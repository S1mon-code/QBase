import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.regime.efficiency_ratio import efficiency_ratio
from indicators.ml.kalman_adaptive import adaptive_kalman
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV260(TimeSeriesStrategy):
    """
    策略简介：自适应Kalman+效率比率双重趋势确认策略。
    使用指标：Adaptive Kalman(60) + Efficiency Ratio(20) + ATR
    进场条件：自适应Kalman上升且效率比率>0.4
    出场条件：ATR追踪止损 / 效率比率跌破0.2
    优点：双重自适应滤波，对市场状态自动调节
    缺点：两个自适应指标叠加可能过度平滑
    """
    name = "mt_v260"
    warmup = 120
    freq = "daily"

    er_period: int = 20
    er_threshold: float = 0.4
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._kalman = None
        self._er = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        self._kalman = adaptive_kalman(closes, period=60)
        self._er = efficiency_ratio(closes, period=self.er_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        kf_val = self._kalman[i]
        er_val = self._er[i]
        if np.isnan(atr_val) or np.isnan(kf_val) or np.isnan(er_val) or atr_val <= 0:
            return
        if i < 1:
            return
        kf_prev = self._kalman[i - 1]
        if np.isnan(kf_prev):
            return
        kf_rising = kf_val > kf_prev

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and kf_rising and er_val > self.er_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and er_val < 0.2:
            context.close_long()
            self._reset()

    def _calc_lots(self, context, price, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_trail_mult * atr_val * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = price * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        return max(1, min(risk_lots, int(context.equity * 0.30 / margin)))

    def _reset(self):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
