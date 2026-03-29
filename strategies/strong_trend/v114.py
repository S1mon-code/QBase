"""
Strong Trend Strategy v114 — Ulcer Index (Low = Calm → Breakout) + Vortex
===========================================================================
Monitors Ulcer Index for periods of calm (low drawdown risk), then enters
when a breakout occurs confirmed by Vortex indicator direction.

  1. Ulcer Index — measures downside volatility/drawdown (low = calm)
  2. Vortex      — directional movement indicator for trend confirmation

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v114.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.ulcer_index import ulcer_index
from indicators.trend.vortex import vortex
from indicators.volatility.atr import atr


class StrongTrendV114(TimeSeriesStrategy):
    """
    策略简介：Ulcer Index低值（平静期）→ Vortex确认突破的策略
    使用指标：Ulcer Index（下行波动率）、Vortex（方向性指标）
    进场条件：Ulcer Index < 历史低位 + Vortex+ > Vortex- + 价格上行
    出场条件：ATR trailing stop 或 Vortex翻空
    优点：Ulcer Index专注下行风险，低值=高性价比入场时机
    缺点：低Ulcer期可能持续很长，需Vortex触发
    """
    name = "strong_trend_v114"
    warmup = 60
    freq = "daily"

    ui_period: int = 14
    vortex_period: int = 14
    atr_trail_mult: float = 4.5
    ui_percentile: float = 30.0

    def __init__(self):
        super().__init__()
        self._ui = None
        self._vortex_plus = None
        self._vortex_minus = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._ui = ulcer_index(closes, period=self.ui_period)
        self._vortex_plus, self._vortex_minus = vortex(highs, lows, closes, period=self.vortex_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._closes = closes

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        if i < 60:
            return

        cur_ui = self._ui[i]
        vp = self._vortex_plus[i]
        vm = self._vortex_minus[i]
        atr_val = self._atr[i]

        if np.isnan(cur_ui) or np.isnan(vp) or np.isnan(vm) or np.isnan(atr_val):
            return

        # Check if Ulcer Index is in low percentile (calm period)
        ui_window = self._ui[max(0, i - 59):i + 1]
        ui_valid = ui_window[~np.isnan(ui_window)]
        if len(ui_valid) < 20:
            return
        ui_thresh = np.percentile(ui_valid, self.ui_percentile)
        ui_calm = cur_ui <= ui_thresh

        vortex_bullish = vp > vm and vp > 1.0

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
        if side == 0 and ui_calm and vortex_bullish:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and vp < vm:
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
