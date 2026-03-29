"""
QBase Strong Trend Strategy v188 — Reflex Indicator + OI Flow
===============================================================

策略简介：Reflex是Ehlers去噪趋势指标，检测价格的真实趋势方向。结合OI Flow确认
         资金流向与趋势一致时做多。

使用指标：
  - Reflex (period=20): Ehlers反射指标，正值=上升趋势
  - OI Flow (period=14): 持仓量资金流
  - ATR (period=14): trailing stop

进场条件：
  1. Reflex > 0（趋势向上）
  2. OI Flow > 0（资金净流入）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. Reflex < -0.5（趋势转下）

优点：Ehlers指标低延迟，OI确认资金面
缺点：Reflex在震荡市频繁翻转
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.momentum.reflex import reflex
from indicators.volume.oi_flow import oi_flow
from indicators.volatility.atr import atr


class StrongTrendV188(TimeSeriesStrategy):
    """Reflex趋势指标 + OI Flow资金流策略。"""
    name = "strong_trend_v188"
    warmup = 60
    freq = "daily"

    reflex_period: int = 20
    oi_flow_period: int = 14
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._reflex = None
        self._oi_flow = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()

        self._reflex = reflex(closes, period=self.reflex_period)
        self._oi_flow = oi_flow(closes, oi, volumes, period=self.oi_flow_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        ref_val = self._reflex[i]
        oif = self._oi_flow[i]
        atr_val = self._atr[i]
        if np.isnan(ref_val) or np.isnan(oif) or np.isnan(atr_val):
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
        if side == 0 and ref_val > 0.0 and oif > 0.0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and ref_val < -0.5:
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
