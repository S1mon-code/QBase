"""
Strong Trend Strategy v15 — EMA Ribbon + CMO + ATR
====================================================
Captures large trending moves (>100%, 6+ months) using:
  1. EMA Ribbon  — trend direction & alignment filter
  2. CMO         — momentum confirmation & strength
  3. ATR         — trailing stop & position sizing

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v15.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v15.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.ema_ribbon import ema_ribbon, ema_ribbon_signal
from indicators.momentum.cmo import cmo
from indicators.volatility.atr import atr


class StrongTrendV15(TimeSeriesStrategy):
    """EMA Ribbon + CMO + ATR — long-only strong trend strategy."""

    name = "strong_trend_v15"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (5) -----
    ribbon_base: int = 8              # Base period; ribbon derived as (base, base*1.6, base*2.6, base*4.2, base*6.9)
    cmo_period: int = 14              # CMO lookback
    cmo_threshold: float = 20.0       # Min CMO to enter
    atr_period: int = 14              # ATR lookback
    atr_trail_mult: float = 3.5       # Trailing stop multiplier

    # ----- Position sizing -----
    contract_multiplier: float = 100.0  # Default for most commodities

    def __init__(self):
        super().__init__()
        self._ribbon_sig = None
        self._ribbons = None      # list of arrays, one per period
        self._cmo = None
        self._atr = None
        self._closes = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0       # 0 = flat, 1-3 = position tiers
        self.trail_stop = 0.0         # Trailing stop price

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ribbon_periods(self) -> tuple[int, ...]:
        """Derive ribbon periods from ribbon_base."""
        b = self.ribbon_base
        return (
            b,
            round(b * 1.6),
            round(b * 2.6),
            round(b * 4.2),
            round(b * 6.9),
        )

    def _update_trail_stop(self, price: float, atr_val: float):
        """Ratchet trailing stop upward (long only)."""
        new_stop = price - self.atr_trail_mult * atr_val
        if new_stop > self.trail_stop:
            self.trail_stop = new_stop

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._closes = closes
        periods = self._ribbon_periods()
        self._ribbon_sig = ema_ribbon_signal(closes, periods)
        self._ribbons = ema_ribbon(closes, periods)
        self._cmo = cmo(closes, self.cmo_period)
        self._atr = atr(highs, lows, closes, self.atr_period)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index

        # Current values
        cur_ribbon = self._ribbon_sig[i]
        cur_cmo = self._cmo[i]
        cur_atr = self._atr[i]
        price = context.close_raw

        # Guard: skip if indicators aren't ready
        if np.isnan(cur_cmo) or np.isnan(cur_atr) or np.isnan(cur_ribbon):
            return

        periods = self._ribbon_periods()

        # Check if close is above all ribbon EMAs
        close_above_all = all(self._closes[i] > self._ribbons[j][i] for j in range(len(periods)))

        # Longest EMA value (for exit check)
        longest_ema = self._ribbons[-1][i]

        side, lots = context.position

        # Update trailing stop if in position
        if lots > 0:
            self._update_trail_stop(price, cur_atr)

        # ==================================================================
        # EXIT — Ribbon fully bearish OR close below longest EMA OR trail stop
        # ==================================================================
        if lots > 0:
            trail_hit = price < self.trail_stop
            ribbon_bearish = cur_ribbon == -1
            below_longest = self._closes[i] < longest_ema

            if ribbon_bearish or below_longest or trail_hit:
                context.close_long()
                self.position_scale = 0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — CMO drops below 0 but ribbon still bullish → close half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_cmo < 0 and cur_ribbon == 1:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — CMO very strong + ribbon bullish + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            if cur_cmo > 50 and cur_ribbon == 1:
                add_lots = self._calc_lots(context, cur_atr)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                return

        # ==================================================================
        # ENTRY — flat, ribbon fully bullish + CMO > threshold + above all EMAs
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            ribbon_bullish = cur_ribbon == 1
            momentum_ok = cur_cmo > self.cmo_threshold
            price_ok = close_above_all

            if ribbon_bullish and momentum_ok and price_ok:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    # Initialize trailing stop
                    self.trail_stop = price - self.atr_trail_mult * cur_atr

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, cur_atr: float) -> int:
        """Size position based on ATR trailing distance.

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
