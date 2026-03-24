"""
Strong Trend Strategy v20 — Fractal Levels + Mass Index + VROC
===============================================================
Fractal breakout with mass index confirming range expansion and VROC
confirming volume participation.

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v20.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v20.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.fractal import fractal_levels
from indicators.trend.mass_index import mass_index
from indicators.volume.vroc import vroc
from indicators.volatility.atr import atr


class StrongTrendV20(TimeSeriesStrategy):
    """Fractal Levels + Mass Index + VROC — long-only strong trend strategy."""

    name = "strong_trend_v20"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (5) -----
    fractal_period: int = 2        # Williams fractal lookback on each side
    mass_ema: int = 9              # Mass Index EMA period
    mass_sum: int = 25             # Mass Index rolling sum period
    vroc_period: int = 14          # Volume Rate of Change lookback
    atr_trail_mult: float = 3.0   # ATR multiplier for trailing stop & sizing

    # ----- Position sizing -----
    contract_multiplier: float = 100.0  # Default for most commodities

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0         # 0 = flat, 1-3 = position tiers
        self.trailing_stop = 0.0        # Trailing stop price
        self.prev_fractal_high = np.nan # Track previous fractal high for add logic
        self.mass_was_above_27 = False  # Track reversal bulge state

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        lookback = max(self.warmup, self.mass_sum + 10, self.vroc_period + 5)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        # ----- Compute indicators -----
        frac_high, frac_low = fractal_levels(highs, lows, self.fractal_period)
        mi = mass_index(highs, lows, self.mass_ema, self.mass_sum)
        vroc_arr = vroc(volumes, self.vroc_period)
        atr_arr = atr(highs, lows, closes, period=14)

        # Current values
        price = context.current_bar.close_raw
        cur_frac_high = frac_high[-1]
        cur_frac_low = frac_low[-1]
        cur_mi = mi[-1]
        cur_vroc = vroc_arr[-1]
        cur_atr = atr_arr[-1]

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_frac_high) or np.isnan(cur_frac_low)
                or np.isnan(cur_mi) or np.isnan(cur_vroc) or np.isnan(cur_atr)):
            return

        side, lots = context.position

        # Track mass index reversal bulge state
        if cur_mi > 27.0:
            self.mass_was_above_27 = True

        # Update trailing stop if long
        if lots > 0 and cur_atr > 0:
            new_stop = price - self.atr_trail_mult * cur_atr
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop

        # ==================================================================
        # EXIT — support broken OR reversal bulge complete OR trailing stop
        # ==================================================================
        if lots > 0:
            support_broken = price < cur_frac_low
            reversal_bulge = self.mass_was_above_27 and cur_mi < 26.5
            trail_hit = price < self.trailing_stop

            if support_broken or reversal_bulge or trail_hit:
                context.close_long()
                self.position_scale = 0
                self.trailing_stop = 0.0
                self.prev_fractal_high = np.nan
                self.mass_was_above_27 = False
                return

        # ==================================================================
        # REDUCE — VROC drying up but still above fractal support
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_vroc < -20.0 and price >= cur_frac_low:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — new higher fractal breakout + VROC positive + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            new_higher_fractal = (
                not np.isnan(self.prev_fractal_high)
                and cur_frac_high > self.prev_fractal_high
            )
            if new_higher_fractal and price > cur_frac_high and cur_vroc > 0:
                add_lots = self._calc_lots(context, cur_atr)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.prev_fractal_high = cur_frac_high
                return

        # ==================================================================
        # ENTRY — flat, breakout above fractal high + mass expansion + VROC
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            breakout = price > cur_frac_high
            mass_expansion = cur_mi > 26.5
            volume_ok = cur_vroc > 0

            if breakout and mass_expansion and volume_ok:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.trailing_stop = price - self.atr_trail_mult * cur_atr
                    self.prev_fractal_high = cur_frac_high
                    self.mass_was_above_27 = cur_mi > 27.0

        # Update prev fractal high tracking even when flat
        if lots == 0:
            self.prev_fractal_high = cur_frac_high

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
        distance = self.atr_trail_mult * cur_atr

        if distance <= 0:
            return 1  # Fallback: minimum lot

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))  # Clamp to [1, 50]
