"""
QBase Strong Trend Strategy v189 — TrendFlex + ADX
=====================================================

策略简介：TrendFlex是Ehlers趋势跟踪指标，正值表示上升趋势。结合ADX确保趋势强度
         足够后入场。

使用指标：
  - TrendFlex (period=20): Ehlers趋势弹性指标
  - ADX (period=14): 趋势强度过滤
  - ATR (period=14): trailing stop

进场条件：
  1. TrendFlex > 0.5（强上升趋势）
  2. ADX > 20（趋势充分发展）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. TrendFlex < 0（趋势消失）

优点：TrendFlex低延迟 + ADX强度确认
缺点：双重趋势过滤可能入场偏晚
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.momentum.trend_flex import trendflex
from indicators.trend.adx import adx
from indicators.volatility.atr import atr


class StrongTrendV189(TimeSeriesStrategy):
    """TrendFlex趋势弹性 + ADX强度策略。"""
    name = "strong_trend_v189"
    warmup = 60
    freq = "daily"

    tf_period: int = 20
    adx_period: int = 14
    adx_entry: float = 20.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._tf = None
        self._adx = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._tf = trendflex(closes, period=self.tf_period)
        self._adx = adx(highs, lows, closes, period=self.adx_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        tf_val = self._tf[i]
        adx_val = self._adx[i]
        atr_val = self._atr[i]
        if np.isnan(tf_val) or np.isnan(adx_val) or np.isnan(atr_val):
            return

        # === Stop Loss Check ===
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # === Entry ===
        if side == 0 and tf_val > 0.5 and adx_val > self.adx_entry:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and tf_val < 0.0:
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
        max_lots = int(context.equity * 0.30 / margin)
        return max(1, min(risk_lots, max_lots))

    def _reset(self):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
