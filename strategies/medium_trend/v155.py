import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.supertrend import supertrend
from indicators.volume.cmf import cmf
from indicators.momentum.ppo import ppo
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV155(TimeSeriesStrategy):
    """
    策略简介：日线Supertrend + 4h CMF资金流 + 10min PPO入场三周期策略。

    使用指标：
    - Supertrend(10, 3.0) [日线]: 大方向
    - CMF(20) [4h]: 中周期资金流>0确认买盘
    - PPO(12,26,9) [10min]: PPO > signal入场
    - ATR(14) [10min]: 止损距离

    进场条件（做多）：日线ST=1 + 4h CMF>0 + 10min PPO>signal且>0
    出场条件：ATR追踪止损, 分层止盈, ST翻转

    优点：量价趋势三维确认，PPO归一化可跨品种
    缺点：CMF可能受极端量能影响
    """
    name = "medium_trend_v155"
    freq = "10min"
    warmup = 3000

    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._ppo_line = None
        self._ppo_signal = None
        self._atr = None
        self._avg_volume = None
        self._st_dir_d = None
        self._cmf_4h = None
        self._d_map = None
        self._4h_map = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        n = len(closes)

        ppo_l, ppo_s, _ = ppo(closes, fast=12, slow=26, signal=9)
        self._ppo_line = ppo_l
        self._ppo_signal = ppo_s
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step_4h = 24
        n_4h = n // step_4h
        trim_4h = n_4h * step_4h
        closes_4h = closes[:trim_4h].reshape(n_4h, step_4h)[:, -1]
        highs_4h = highs[:trim_4h].reshape(n_4h, step_4h).max(axis=1)
        lows_4h = lows[:trim_4h].reshape(n_4h, step_4h).min(axis=1)
        volumes_4h = volumes[:trim_4h].reshape(n_4h, step_4h).sum(axis=1)
        self._cmf_4h = cmf(highs_4h, lows_4h, closes_4h, volumes_4h, period=20)
        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step_4h - 1), n_4h - 1)

        _, self._st_dir_d = supertrend(highs_4h, lows_4h, closes_4h, period=10, multiplier=3.0)
        self._d_map = self._4h_map

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        ppo_v = self._ppo_line[i]
        ppo_s = self._ppo_signal[i]
        atr_val = self._atr[i]
        sd = self._st_dir_d[j]
        cmf_val = self._cmf_4h[j]
        if np.isnan(ppo_v) or np.isnan(ppo_s) or np.isnan(atr_val) or np.isnan(sd) or np.isnan(cmf_val):
            return

        daily_up = sd == 1
        money_in = cmf_val > 0
        self.bars_since_last_scale += 1

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

        if side == 1 and self.entry_price > 0:
            profit_atr = (price - self.entry_price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_long(lots=max(1, lots // 3))
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_long(lots=max(1, lots // 3))
                self._took_profit_3atr = True
                return

        if side == 1 and not daily_up:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and daily_up and money_in and ppo_v > ppo_s and ppo_v > 0:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and daily_up and money_in):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _calc_lots(self, context, atr_val):
        from alphaforge.data.contract_specs import ContractSpecManager
        spec = ContractSpecManager().get(context.symbol)
        stop_dist = self.atr_stop_mult * atr_val * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = context.close_raw * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        return max(1, min(risk_lots, int(context.equity * 0.30 / margin)))

    def _reset_state(self):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
