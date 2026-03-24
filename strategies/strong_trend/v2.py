"""
Strong Trend Strategy v2 — KAMA + R-Squared + ATR (Trend Quality)
=================================================================
Captures strong trends by combining adaptive price following (KAMA),
trend quality measurement (R²), and volatility-based risk management (ATR).

LONG ONLY — position scaling 0-3.

Usage:
    ./run.sh strategies/strong_trend/v2.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# Ensure QBase root is in path for indicator imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.momentum.kama import kama
from indicators.trend.linear_regression import r_squared
from indicators.volatility.atr import atr


class StrongTrendV2(TimeSeriesStrategy):
    """KAMA + R² + ATR trend-quality strategy (long only, scale 0-3)."""

    name = "strong_trend_v2"
    warmup = 60
    freq = "daily"

    # ── Parameters (5 total) ──────────────────────────────────────────
    kama_period = 10        # KAMA lookback
    rsq_period = 40         # R-squared lookback
    rsq_threshold = 0.5     # Minimum R² for entry
    atr_period = 14         # ATR lookback
    trail_mult = 3.5        # Trailing stop multiplier

    # ── Internal state ────────────────────────────────────────────────

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0       # Current scale level (0-3)
        self.highest_since_entry = 0.0  # Highest price since initial entry
        self.entry_price = 0.0        # Price at initial entry

    def on_bar(self, context):
        """Core strategy logic — called every bar after warmup."""
        # -- Fetch price arrays (enough history for all indicators) --
        lookback = max(self.warmup, self.rsq_period + 10, self.atr_period * 2 + 10)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)

        price = context.current_bar.close_raw
        side, lots = context.position

        # -- Compute indicators --
        kama_arr = kama(closes, period=self.kama_period)
        rsq_arr = r_squared(closes, period=self.rsq_period)
        atr_arr = atr(highs, lows, closes, period=self.atr_period)

        # Current values
        kama_now = kama_arr[-1]
        kama_prev = kama_arr[-2]
        rsq_now = rsq_arr[-1]
        atr_now = atr_arr[-1]

        # Mean ATR over 2x period for volatility filter
        atr_window = atr_arr[-(self.atr_period * 2):]
        atr_mean = np.nanmean(atr_window)

        # -- Guard: skip if any indicator not ready --
        if np.isnan(kama_now) or np.isnan(kama_prev) or np.isnan(rsq_now) or np.isnan(atr_now):
            return
        if np.isnan(atr_mean) or atr_now <= 0:
            return

        # -- Contract multiplier for position sizing --
        contract_mult = getattr(context, "contract_multiplier", 100)

        # ═══════════════════════════════════════════════════════════════
        # NO POSITION — look for entry
        # ═══════════════════════════════════════════════════════════════
        if lots == 0:
            kama_rising = kama_now > kama_prev
            above_kama = closes[-1] > kama_now
            trend_quality = rsq_now > self.rsq_threshold
            vol_expanding = atr_now >= atr_mean

            if above_kama and kama_rising and trend_quality and vol_expanding:
                lot_size = self._calc_lots(context, atr_now, contract_mult)
                if lot_size > 0:
                    context.buy(lot_size)
                    self.position_scale = 1
                    self.entry_price = price
                    self.highest_since_entry = price
            return

        # ═══════════════════════════════════════════════════════════════
        # IN POSITION (long only) — manage adds, reduces, exits
        # ═══════════════════════════════════════════════════════════════
        if side != 1:
            return  # Safety: should never happen (long only)

        # Update highest price since entry
        self.highest_since_entry = max(self.highest_since_entry, price)

        # -- Trailing stop level --
        trail_stop = self.highest_since_entry - self.trail_mult * atr_now

        # ── EXIT: trailing stop hit OR R² collapsed ──────────────────
        if price < trail_stop or rsq_now < 0.3:
            context.close_long()
            self.position_scale = 0
            self.highest_since_entry = 0.0
            self.entry_price = 0.0
            return

        # ── REDUCE: R² weakening (0.3-0.5 range) ────────────────────
        if 0.3 <= rsq_now < 0.5 and self.position_scale > 1:
            half_lots = max(1, lots // 2)
            context.close_long(lots=half_lots)
            self.position_scale -= 1
            return

        # ── ADD: R² strong AND price makes new high since entry ──────
        if (rsq_now > 0.6
                and price >= self.highest_since_entry
                and self.position_scale < 3):
            add_lots = self._calc_lots(context, atr_now, contract_mult)
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1

    # ── Position sizing ───────────────────────────────────────────────
    def _calc_lots(self, context, atr_now, contract_mult):
        """Risk-based sizing: risk 2% of equity per unit of risk."""
        risk_budget = context.equity * 0.02
        risk_per_lot = self.trail_mult * atr_now * contract_mult

        if risk_per_lot <= 0:
            return 1

        lot_size = int(risk_budget / risk_per_lot)
        return max(1, min(lot_size, 50))
