import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.ml.ensemble_signal import ensemble_vote
from indicators.trend.supertrend import supertrend
from indicators.momentum.rsi import rsi
from indicators.volatility.atr import atr
from indicators.trend.adx import adx
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV160(TimeSeriesStrategy):
    """
    策略简介：日线Ensemble ML投票 + 4h Supertrend + 5min RSI三周期策略。

    使用指标：
    - Ensemble Vote [日线]: 多模型投票信号>0.5确认大趋势
    - Supertrend(10, 3.0) [4h]: 中周期趋势方向
    - RSI(14) [5min]: 超卖回调入场
    - ATR(14) [5min]: 止损距离

    进场条件（做多）：日线ensemble>0.5 + 4h ST=1 + 5min RSI<30
    出场条件：ATR追踪止损, 分层止盈, ensemble<0或ST翻转

    优点：Ensemble集成多模型降低单一模型风险
    缺点：Ensemble计算最重，可能过拟合特征
    """
    name = "medium_trend_v160"
    freq = "5min"
    warmup = 3000

    rsi_entry: float = 30.0
    ensemble_threshold: float = 0.5
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._rsi = None
        self._atr = None
        self._avg_volume = None
        self._ensemble_d = None
        self._st_dir_4h = None
        self._d_map = None

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

        self._rsi = rsi(closes, 14)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 48
        nd = n // step
        trim = nd * step
        closes_d = closes[:trim].reshape(nd, step)[:, -1]
        highs_d = highs[:trim].reshape(nd, step).max(axis=1)
        lows_d = lows[:trim].reshape(nd, step).min(axis=1)

        # Build features for ensemble
        adx_d = adx(highs_d, lows_d, closes_d, period=14)
        rsi_d = rsi(closes_d, 14)
        returns_d = np.zeros_like(closes_d)
        returns_d[1:] = np.diff(closes_d) / closes_d[:-1]
        features = np.column_stack([adx_d, rsi_d, returns_d])
        # Replace NaN with 0 for ensemble
        features = np.nan_to_num(features, nan=0.0)

        self._ensemble_d = ensemble_vote(closes_d, features, period=120)
        _, self._st_dir_4h = supertrend(highs_d, lows_d, closes_d, period=10, multiplier=3.0)
        self._d_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1), nd - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._d_map[i]
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        rsi_val = self._rsi[i]
        atr_val = self._atr[i]
        ens = self._ensemble_d[j]
        sd = self._st_dir_4h[j]
        if np.isnan(rsi_val) or np.isnan(atr_val) or np.isnan(ens) or np.isnan(sd):
            return

        ens_bull = ens > self.ensemble_threshold
        daily_up = sd == 1
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

        if side == 1 and (ens < 0 or not daily_up):
            context.close_long()
            self._reset_state()
            return

        if side == 0 and ens_bull and daily_up and rsi_val < self.rsi_entry:
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
                    and ens_bull and daily_up):
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
