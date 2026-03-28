import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.ml.trend_filter import trend_filter
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV257(TimeSeriesStrategy):
    """
    策略简介：L1趋势滤波（Hodrick-Prescott变体），提取分段线性趋势。
    使用指标：Trend Filter(60, lambda=1.0) + ATR
    进场条件：滤波趋势持续上升且价格在滤波值上方
    出场条件：ATR追踪止损 / 滤波趋势拐头向下
    优点：分段线性拟合更贴合实际趋势形态
    缺点：lambda参数选择影响大
    """
    name = "mt_v257"
    warmup = 120
    freq = "daily"

    tf_period: int = 60
    tf_lambda: float = 1.0
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._tf = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        self._tf = trend_filter(closes, period=self.tf_period, lambda_param=self.tf_lambda)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        tf_val = self._tf[i]
        if np.isnan(atr_val) or np.isnan(tf_val) or atr_val <= 0:
            return
        if i < 3:
            return
        tf_prev = self._tf[i - 1]
        tf_prev2 = self._tf[i - 3]
        if np.isnan(tf_prev) or np.isnan(tf_prev2):
            return
        trend_up = tf_val > tf_prev and tf_prev > tf_prev2

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and trend_up and price > tf_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and tf_val < tf_prev:
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
