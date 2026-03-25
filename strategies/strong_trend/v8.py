"""
Strong Trend Strategy v8 — Linear Regression Slope + Choppiness Index + NATR
=============================================================================
Captures large trending moves using:
  1. Linear Regression Slope — trend direction & strength
  2. Choppiness Index        — trending vs choppy regime filter
  3. NATR                    — volatility regime filter
  4. ATR                     — trailing stop & position sizing

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v8.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v8.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.linear_regression import linear_regression_slope
from indicators.momentum.chop_zone import choppiness_index
from indicators.volatility.natr import natr
from indicators.volatility.atr import atr


class StrongTrendV8(TimeSeriesStrategy):
    """LinReg Slope + Choppiness Index + NATR — long-only strong trend strategy."""

    name = "strong_trend_v8"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (5) -----
    slope_period: int = 20          # Linear regression slope lookback
    chop_period: int = 14           # Choppiness Index lookback
    chop_threshold: float = 50.0    # Max choppiness for entry (lower = trending)
    natr_period: int = 14           # NATR lookback
    atr_trail_mult: float = 3.5     # ATR multiplier for trailing stop & sizing

    # ----- Position sizing -----
    contract_multiplier: float = 100.0  # Default for most commodities

    def __init__(self):
        super().__init__()
        self._slope = None
        self._chop = None
        self._natr = None
        self._atr = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0         # 0 = flat, 1-3 = position tiers
        self.prev_slope = np.nan        # Previous bar's slope for acceleration
        self.trail_stop = 0.0           # Trailing stop level

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._slope = linear_regression_slope(closes, self.slope_period)
        self._chop = choppiness_index(highs, lows, closes, self.chop_period)
        self._natr = natr(highs, lows, closes, self.natr_period)
        self._atr = atr(highs, lows, closes, period=14)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index

        if i < 1:
            return

        # ----- Look up pre-computed indicators -----
        cur_slope = self._slope[i]
        prev_slope_ind = self._slope[i - 1]
        cur_chop = self._chop[i]
        cur_natr = self._natr[i]
        cur_atr = self._atr[i]
        price = context.close_raw

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_slope) or np.isnan(cur_chop)
                or np.isnan(cur_natr) or np.isnan(cur_atr)):
            self.prev_slope = cur_slope
            return

        side, lots = context.position

        # ----- Update trailing stop -----
        if lots > 0 and cur_atr > 0:
            new_stop = price - self.atr_trail_mult * cur_atr
            self.trail_stop = max(self.trail_stop, new_stop)

        # ==================================================================
        # EXIT — Slope negative OR Choppiness > 70 OR trailing stop hit
        # ==================================================================
        if lots > 0:
            slope_exit = cur_slope < 0
            chop_exit = cur_chop > 70.0
            trail_exit = price <= self.trail_stop

            if slope_exit or chop_exit or trail_exit:
                context.close_long()
                self.position_scale = 0
                self.trail_stop = 0.0
                self.prev_slope = cur_slope
                return

        # ==================================================================
        # REDUCE — Choppiness rising above 60 but slope still positive
        # Close half the position when regime starts getting choppy
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_chop > 60.0 and cur_slope > 0:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                self.prev_slope = cur_slope
                return

        # ==================================================================
        # ADD — Already long, slope increasing, choppiness still low
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            slope_increasing = (
                not np.isnan(self.prev_slope)
                and cur_slope > self.prev_slope
                and cur_slope > 0
            )
            chop_still_low = cur_chop < self.chop_threshold

            if slope_increasing and chop_still_low:
                add_lots = self._calc_lots(context, cur_atr)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                self.prev_slope = cur_slope
                return

        # ==================================================================
        # ENTRY — Flat, slope positive, low choppiness, reasonable NATR
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            trend_up = cur_slope > 0
            trending_regime = cur_chop < self.chop_threshold
            # NATR filter: avoid extremely low volatility (dead market)
            vol_ok = cur_natr > 0.5

            if trend_up and trending_regime and vol_ok:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    # Initialize trailing stop on entry
                    self.trail_stop = price - self.atr_trail_mult * cur_atr

        # Save slope for next bar's acceleration check
        self.prev_slope = cur_slope

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, cur_atr: float) -> int:
        """Size position based on ATR trailing stop distance.

        lots = (equity * 0.02) / (atr_trail_mult * ATR * contract_multiplier)
        Ensures each lot risks roughly 2% of equity if price hits the
        trailing stop level.
        """
        risk_per_trade = context.equity * 0.02
        risk_distance = self.atr_trail_mult * cur_atr

        if risk_distance <= 0:
            return 1  # Fallback: minimum lot

        risk_per_lot = risk_distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))  # Clamp to [1, 50]
