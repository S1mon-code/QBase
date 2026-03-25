"""
Strong Trend Strategy v13 — ZLEMA + Fisher Transform + Range Expansion
======================================================================
Captures large trending moves (>100%, 6+ months) using:
  1. ZLEMA             — zero-lag trend direction & trailing reference
  2. Fisher Transform  — momentum confirmation (bullish cross)
  3. Range Expansion   — volatility expansion filter

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v13.py --symbols AG --freq 4h --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v13.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.zlema import zlema
from indicators.momentum.fisher_transform import fisher_transform
from indicators.volatility.range_expansion import range_expansion
from indicators.volatility.atr import atr


class StrongTrendV13(TimeSeriesStrategy):
    """ZLEMA + Fisher Transform + Range Expansion — long-only strong trend strategy."""

    name = "strong_trend_v13"
    warmup = 60
    freq = "4h"

    # ----- Tunable parameters (5) -----
    zlema_period: int = 20           # ZLEMA lookback
    fisher_period: int = 10          # Fisher Transform lookback
    re_period: int = 14              # Range Expansion lookback
    re_threshold: float = 1.2        # Min range expansion ratio for entry
    atr_trail_mult: float = 3.0      # ATR multiplier for trailing stop & sizing

    # ----- Position sizing -----
    contract_multiplier: float = 100.0  # Default for most commodities

    def __init__(self):
        super().__init__()
        self._zlema = None
        self._fisher = None
        self._trigger = None
        self._re = None
        self._atr = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0      # 0 = flat, 1-3 = position tiers
        self.trail_stop = 0.0        # Trailing stop price

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._zlema = zlema(closes, self.zlema_period)
        self._fisher, self._trigger = fisher_transform(highs, lows, self.fisher_period)
        self._re = range_expansion(highs, lows, closes, self.re_period)
        self._atr = atr(highs, lows, closes, period=14)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index
        if i < 1:
            return

        # Current values
        price = context.close_raw
        cur_zl = self._zlema[i]
        prev_zl = self._zlema[i - 1]
        cur_fisher = self._fisher[i]
        cur_trigger = self._trigger[i]
        cur_re = self._re[i]
        cur_atr = self._atr[i]

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_zl) or np.isnan(prev_zl) or np.isnan(cur_fisher)
                or np.isnan(cur_trigger) or np.isnan(cur_re) or np.isnan(cur_atr)):
            return

        # Derived conditions
        zlema_rising = cur_zl > prev_zl
        above_zlema = price > cur_zl
        fisher_bullish_cross = cur_fisher > cur_trigger
        re_expanding = cur_re > self.re_threshold

        side, lots = context.position

        # ==================================================================
        # TRAILING STOP — update & check
        # ==================================================================
        if lots > 0 and cur_atr > 0:
            new_stop = price - self.atr_trail_mult * cur_atr
            if new_stop > self.trail_stop:
                self.trail_stop = new_stop

            if price <= self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # EXIT — close below ZLEMA OR Fisher crosses below trigger
        # ==================================================================
        if lots > 0:
            if not above_zlema or cur_fisher < cur_trigger:
                context.close_long()
                self.position_scale = 0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — Range expansion contracting but ZLEMA still rising
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_re < 1.0 and zlema_rising:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — Fisher strongly bullish + Range expansion still high
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            if cur_fisher > 1.0 and cur_re > self.re_threshold:
                add_lots = self._calc_lots(context, cur_atr)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                return

        # ==================================================================
        # ENTRY — flat, all conditions aligned
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            if above_zlema and zlema_rising and fisher_bullish_cross and re_expanding:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.trail_stop = price - self.atr_trail_mult * cur_atr

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per ATR-based stop distance
    # ------------------------------------------------------------------
    def _calc_lots(self, context, cur_atr: float) -> int:
        """Size position based on ATR trailing stop distance.

        lots = (equity * 0.02) / (atr_trail_mult * ATR * contract_multiplier)
        Ensures each lot risks roughly 2% of equity if price hits the
        trailing stop level.
        """
        risk_per_trade = context.equity * 0.02
        distance = self.atr_trail_mult * cur_atr

        if distance <= 0:
            return 1  # Fallback: minimum lot

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))  # Clamp to [1, 50]
