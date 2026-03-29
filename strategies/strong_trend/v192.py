"""
QBase Strong Trend Strategy v192 — Center of Gravity + Vortex
===============================================================

策略简介：Center of Gravity是Ehlers低延迟振荡器，结合Vortex指标确认趋势方向。
         COG上穿零线且Vortex VI+>VI-时做多。

使用指标：
  - Center of Gravity (period=10): Ehlers重心振荡器
  - Vortex (period=14): VI+/VI-趋势方向
  - ATR (period=14): trailing stop

进场条件：
  1. COG > 0（价格在重心上方）
  2. VI+ > VI-（正向涡旋主导）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. VI- > VI+（负向涡旋主导）

优点：COG极低延迟，Vortex方向明确
缺点：COG在震荡市频繁翻转
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.momentum.center_of_gravity import cog
from indicators.trend.vortex import vortex
from indicators.volatility.atr import atr


class StrongTrendV192(TimeSeriesStrategy):
    """重心振荡器 + Vortex方向策略。"""
    name = "strong_trend_v192"
    warmup = 60
    freq = "daily"

    cog_period: int = 10
    vortex_period: int = 14
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._cog = None
        self._vi_plus = None
        self._vi_minus = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._cog = cog(closes, period=self.cog_period)
        self._vi_plus, self._vi_minus = vortex(highs, lows, closes, period=self.vortex_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        cog_val = self._cog[i]
        vip = self._vi_plus[i]
        vim = self._vi_minus[i]
        atr_val = self._atr[i]
        if np.isnan(cog_val) or np.isnan(vip) or np.isnan(vim) or np.isnan(atr_val):
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
        if side == 0 and cog_val > 0.0 and vip > vim:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and vim > vip:
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
