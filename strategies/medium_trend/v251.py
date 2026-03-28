import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.ml.kalman_trend import kalman_filter
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV251(TimeSeriesStrategy):
    """
    策略简介：Kalman滤波趋势跟踪，利用卡尔曼滤波器提取平滑趋势方向。
    使用指标：Kalman Filter + ATR
    进场条件：价格持续高于Kalman趋势线且趋势斜率为正
    出场条件：ATR追踪止损 / 价格跌破Kalman趋势线
    优点：自适应平滑，噪声过滤能力强
    缺点：参数敏感，快速反转时滞后
    """
    name = "mt_v251"
    warmup = 120
    freq = "daily"

    process_noise: float = 0.005
    measurement_noise: float = 1.0
    slope_lookback: int = 5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._kalman = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        self._kalman = kalman_filter(closes, process_noise=self.process_noise, measurement_noise=self.measurement_noise)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        kf_val = self._kalman[i]
        if np.isnan(atr_val) or np.isnan(kf_val) or atr_val <= 0:
            return
        if i < self.slope_lookback:
            return
        kf_prev = self._kalman[i - self.slope_lookback]
        if np.isnan(kf_prev):
            return
        slope_positive = kf_val > kf_prev

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and price > kf_val and slope_positive:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and price < kf_val:
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
