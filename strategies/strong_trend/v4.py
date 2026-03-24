"""
Strong Trend Strategy v4 — EMA Cross + MACD + ATR
===================================================
Captures large trending moves (>100%, 6 months+) using:
  1. EMA crossover (trend direction)
  2. MACD histogram (momentum confirmation & scaling)
  3. ATR (trailing stop & position sizing)

Long-only. Supports position scaling 0-3.

Usage:
    ./run.sh strategies/strong_trend/v4.py --symbols AG --freq 4h --start 2022
"""
import sys
from pathlib import Path

# QBase root on sys.path so indicators and conftest resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401  — wires up AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.ema import ema_cross
from indicators.momentum.macd import macd
from indicators.volatility.atr import atr


class StrongTrendV4(TimeSeriesStrategy):
    name = "strong_trend_v4"
    warmup = 80
    freq = "4h"

    # ── Tunable parameters (5) ──────────────────────────────────────────
    ema_fast: int = 20
    ema_slow: int = 60
    macd_fast: int = 12
    macd_slow: int = 26
    atr_trail_mult: float = 3.0

    # Internal (not tunable)
    _atr_period: int = 14

    # ── Lifecycle ────────────────────────────────────────────────────────

    def on_init(self, context):
        """Reset per-run tracking state."""
        self.highest_since_entry = 0.0
        self.position_scale = 0          # 0-3 scale tracker
        self.prev_macd_hist = np.nan     # for histogram acceleration check

    # ── Core logic ───────────────────────────────────────────────────────

    def on_bar(self, context):
        """Called every bar — entry, add, reduce, and exit logic."""
        # Need enough bars for the slowest indicator (EMA 60 + some margin)
        n = self.warmup
        closes = context.get_close_array(n)
        highs = context.get_high_array(n)
        lows = context.get_low_array(n)

        if len(closes) < n:
            return

        price = context.current_bar.close_raw
        side, lots = context.position

        # ── Compute indicators ───────────────────────────────────────────
        fast_ema, slow_ema, cross_signal = ema_cross(
            closes, self.ema_fast, self.ema_slow,
        )
        _, _, histogram = macd(closes, self.macd_fast, self.macd_slow)
        atr_values = atr(highs, lows, closes, period=self._atr_period)

        # Current values
        cross_now = cross_signal[-1]         # +1 golden, -1 death, 0 none
        fast_now = fast_ema[-1]
        slow_now = slow_ema[-1]
        hist_now = histogram[-1]
        hist_prev = histogram[-2]
        atr_now = atr_values[-1]

        # Bail if any indicator is still warming up
        if np.isnan(hist_now) or np.isnan(hist_prev) or np.isnan(atr_now):
            return

        # ── Update trailing-stop high watermark ──────────────────────────
        if side == 1:
            if price > self.highest_since_entry:
                self.highest_since_entry = price

        # ── EXIT: Death cross (EMA fast crosses below slow) ─────────────
        if side == 1 and cross_now == -1.0:
            context.close_long()
            self._reset_state()
            return

        # ── EXIT: Trailing stop ──────────────────────────────────────────
        if side == 1:
            trail_stop = self.highest_since_entry - self.atr_trail_mult * atr_now
            if price <= trail_stop:
                context.close_long()
                self._reset_state()
                return

        # ── REDUCE: MACD histogram turns negative, but EMA still bullish ─
        if side == 1 and hist_now < 0 and fast_now > slow_now:
            half = max(1, lots // 2)
            context.close_long(lots=half)
            self.position_scale = max(0, self.position_scale - 1)
            return

        # ── ADD: MACD histogram accelerating + already long + scale < 3 ──
        if side == 1 and self.position_scale < 3:
            if hist_now > hist_prev and hist_now > 0:
                add_lots = self._calc_lots(context, atr_now)
                if add_lots >= 1:
                    context.buy(add_lots)
                    self.position_scale += 1
                return

        # ── ENTRY: Golden cross + MACD histogram > 0 ────────────────────
        if lots == 0 and cross_now == 1.0 and hist_now > 0:
            entry_lots = self._calc_lots(context, atr_now)
            if entry_lots >= 1:
                context.buy(entry_lots)
                self.highest_since_entry = price
                self.position_scale = 1
            return

        # Store previous histogram for next bar's acceleration check
        self.prev_macd_hist = hist_now

    # ── Helpers ──────────────────────────────────────────────────────────

    def _calc_lots(self, context, atr_now: float) -> int:
        """Position sizing: risk 2% of equity per unit of ATR-based stop.

        lots = equity * 0.02 / (atr_trail_mult * ATR * contract_multiplier)
        """
        contract_multiplier = getattr(context, "contract_multiplier", 1)
        risk_per_lot = self.atr_trail_mult * atr_now * contract_multiplier
        if risk_per_lot <= 0:
            return 1
        raw = context.equity * 0.02 / risk_per_lot
        return max(1, int(raw))

    def _reset_state(self):
        """Clear tracking state after a full exit."""
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.prev_macd_hist = np.nan
