"""
Strong Trend Strategy v11 — Vortex + ROC + OI Momentum
=======================================================
Captures large trending moves (>100%, 6+ months) using:
  1. Vortex    — trend direction (VI+ vs VI-)
  2. ROC       — momentum confirmation & acceleration
  3. OI Momentum — conviction filter (new money entering/leaving)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v11.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v11.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.vortex import vortex
from indicators.momentum.roc import rate_of_change
from indicators.volume.oi_momentum import oi_momentum
from indicators.volatility.atr import atr


class StrongTrendV11(TimeSeriesStrategy):
    """Vortex + ROC + OI Momentum — long-only strong trend strategy."""

    name = "strong_trend_v11"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (5) -----
    vortex_period: int = 14       # Vortex Indicator lookback
    roc_period: int = 20          # Rate of Change lookback
    roc_threshold: float = 5.0    # Min ROC% to enter
    oi_period: int = 20           # OI Momentum lookback
    atr_trail_mult: float = 3.0   # ATR multiplier for trailing stop

    # ----- Position sizing -----
    contract_multiplier: float = 100.0  # Default for most commodities

    def __init__(self):
        super().__init__()
        self._vi_plus = None
        self._vi_minus = None
        self._roc = None
        self._oi_mom = None
        self._atr = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0     # 0 = flat, 1-3 = position tiers
        self.prev_roc = np.nan      # Previous bar's ROC for acceleration check
        self.prev_vi_spread = np.nan  # Previous VI+ - VI- for widening check
        self.trail_stop = 0.0       # Trailing stop price
        self.highest_since_entry = 0.0  # Track highest price for trailing stop

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._vi_plus, self._vi_minus = vortex(highs, lows, closes, self.vortex_period)
        self._roc = rate_of_change(closes, self.roc_period)
        self._atr = atr(highs, lows, closes, period=14)

        # OI momentum — use volume as OI proxy if OI not available
        try:
            oi_arr = context.get_full_oi_array()
        except (AttributeError, Exception):
            oi_arr = volumes  # Fallback: use volume as proxy
        self._oi_mom = oi_momentum(oi_arr, self.oi_period)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index

        cur_vi_plus = self._vi_plus[i]
        cur_vi_minus = self._vi_minus[i]
        cur_roc = self._roc[i]
        cur_oi_mom = self._oi_mom[i]
        cur_atr = self._atr[i]
        price = context.close_raw

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_vi_plus) or np.isnan(cur_vi_minus) or
                np.isnan(cur_roc) or np.isnan(cur_oi_mom) or np.isnan(cur_atr)):
            self.prev_roc = cur_roc
            self.prev_vi_spread = cur_vi_plus - cur_vi_minus if not (
                np.isnan(cur_vi_plus) or np.isnan(cur_vi_minus)) else np.nan
            return

        vi_spread = cur_vi_plus - cur_vi_minus
        side, lots = context.position

        # Update trailing stop for existing long positions
        if lots > 0:
            self.highest_since_entry = max(self.highest_since_entry, price)
            self.trail_stop = self.highest_since_entry - self.atr_trail_mult * cur_atr

        # ==================================================================
        # EXIT — VI- > VI+ (trend reversal) OR trailing stop hit → close all
        # ==================================================================
        if lots > 0:
            trend_reversed = cur_vi_minus > cur_vi_plus
            stop_hit = price < self.trail_stop

            if trend_reversed or stop_hit:
                context.close_long()
                self.position_scale = 0
                self.trail_stop = 0.0
                self.highest_since_entry = 0.0
                self.prev_roc = cur_roc
                self.prev_vi_spread = vi_spread
                return

        # ==================================================================
        # REDUCE — OI momentum turns negative (money leaving) but VI+ > VI-
        # Close half the position
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_oi_mom < 0 and cur_vi_plus > cur_vi_minus:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                self.prev_roc = cur_roc
                self.prev_vi_spread = vi_spread
                return

        # ==================================================================
        # ADD — VI spread widening + ROC accelerating + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            spread_widening = (
                not np.isnan(self.prev_vi_spread)
                and vi_spread > self.prev_vi_spread
                and vi_spread > 0
            )
            roc_accelerating = (
                not np.isnan(self.prev_roc)
                and self.prev_roc > 0
                and cur_roc > self.prev_roc
            )

            if spread_widening and roc_accelerating:
                add_lots = self._calc_lots(context, cur_atr)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                self.prev_roc = cur_roc
                self.prev_vi_spread = vi_spread
                return

        # ==================================================================
        # ENTRY — flat, VI+ > VI-, strong momentum, new money entering
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            trend_bullish = cur_vi_plus > cur_vi_minus
            momentum_ok = cur_roc > self.roc_threshold
            new_money = cur_oi_mom > 0

            if trend_bullish and momentum_ok and new_money:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.highest_since_entry = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr

        # Save state for next bar
        self.prev_roc = cur_roc
        self.prev_vi_spread = vi_spread

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
        stop_distance = self.atr_trail_mult * cur_atr

        if stop_distance <= 0:
            return 1  # Fallback: minimum lot

        risk_per_lot = stop_distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))  # Clamp to [1, 50]
