import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.trend.supertrend import supertrend
from indicators.trend.hma import hma
from indicators.momentum.stochastic import stochastic
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV143(TimeSeriesStrategy):
    """
    策略简介：日线Supertrend + 4h HMA + 5min Stochastic三周期策略。

    使用指标：
    - Supertrend(10, 3.0) [日线]: 大周期趋势方向
    - HMA(20) [4h]: 中周期趋势斜率确认
    - Stochastic %K(14) [5min]: 小周期超卖入场
    - ATR(14) [5min]: 止损距离

    进场条件（做多）：日线ST=1 + 4h HMA上升 + 5min %K<20
    出场条件：ATR追踪止损, 分层止盈, ST翻转

    优点：HMA中间层平滑过渡，三重确认
    缺点：条件严格，入场机会少
    """
    name = "medium_trend_v143"
    freq = "5min"
    warmup = 3000

    hma_period: int = 20
    stoch_entry: float = 20.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._stoch_k = None
        self._atr = None
        self._avg_volume = None
        self._st_dir = None
        self._hma_4h = None
        self._map = None

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

        k_arr, _ = stochastic(highs, lows, closes, k=14, d=3)
        self._stoch_k = k_arr
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 48
        n_big = n // step
        trim = n_big * step
        closes_big = closes[:trim].reshape(n_big, step)[:, -1]
        highs_big = highs[:trim].reshape(n_big, step).max(axis=1)
        lows_big = lows[:trim].reshape(n_big, step).min(axis=1)

        _, self._st_dir = supertrend(highs_big, lows_big, closes_big, period=10, multiplier=3.0)
        self._hma_4h = hma(closes_big, period=self.hma_period)
        self._map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1), n_big - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._map[i]
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        sk = self._stoch_k[i]
        atr_val = self._atr[i]
        sd = self._st_dir[j]
        h_cur = self._hma_4h[j]
        h_prev = self._hma_4h[j - 1] if j > 0 else np.nan
        if np.isnan(sk) or np.isnan(atr_val) or np.isnan(sd) or np.isnan(h_cur) or np.isnan(h_prev):
            return

        daily_up = (sd == 1)
        hma_rising = (h_cur > h_prev)
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

        if side == 0 and daily_up and hma_rising and sk < self.stoch_entry:
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
                    and daily_up and hma_rising):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
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
