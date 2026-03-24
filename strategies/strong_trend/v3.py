"""
Strong Trend v3 — Donchian Channel + ADX + Chandelier Exit (Classic Turtle)
============================================================================
Captures large trending moves (>100% rallies, 6+ months) using a Donchian
channel breakout for entry, ADX for trend strength confirmation, and the
Chandelier Exit as an ATR-based trailing stop.

LONG ONLY. Supports position scaling 0-3.

Usage:
    ./run.sh strategies/strong_trend/v3.py --symbols AG --freq daily --start 2022

Parameters (5):
    don_period    = 55   Donchian channel lookback
    adx_period    = 14   ADX smoothing period
    adx_threshold = 25   Minimum ADX to confirm trend strength
    chand_period  = 22   Chandelier Exit lookback
    chand_mult    = 3.0  Chandelier ATR multiplier
"""
import sys
from pathlib import Path

# Add QBase root to path (two levels up: strong_trend -> strategies -> QBase)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.donchian import donchian
from indicators.trend.adx import adx
from indicators.volatility.chandelier_exit import chandelier_exit


class DonchianADXChandelierStrategy(TimeSeriesStrategy):
    name = "strong_trend_v3_turtle"
    warmup = 60
    freq = "daily"

    # ── Tunable parameters (5 total) ──────────────────────────────────
    don_period: int = 55
    adx_period: int = 14
    adx_threshold: int = 25
    chand_period: int = 22
    chand_mult: float = 3.0

    # ── Internal constants ────────────────────────────────────────────
    _ADX_WEAK_LOW = 20   # Lower bound of "weakening trend" zone
    _MAX_SCALE = 3       # Maximum position scale level

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0        # Current scale level (0 = flat, 1-3 = sized)
        self.last_breakout_high = 0.0  # Track the Donchian upper at last entry/add

    def on_bar(self, context):
        """Core strategy logic — called on every bar after warmup."""
        # Need enough bars for the longest lookback indicator
        lookback = max(self.don_period, self.chand_period) + self.adx_period + 10
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)

        if len(closes) < lookback:
            return

        price = context.current_bar.close_raw

        # ── Compute indicators ────────────────────────────────────────
        don_upper, _, _ = donchian(highs, lows, period=self.don_period)
        adx_vals = adx(highs, lows, closes, period=self.adx_period)
        chand_long_exit, _ = chandelier_exit(
            highs, lows, closes,
            period=self.chand_period,
            multiplier=self.chand_mult,
        )

        # Current values (latest bar)
        don_upper_now = don_upper[-1]
        adx_now = adx_vals[-1]
        chand_long_now = chand_long_exit[-1]

        # Skip if any indicator is not yet valid
        if np.isnan(don_upper_now) or np.isnan(adx_now) or np.isnan(chand_long_now):
            return

        side, lots = context.position

        # ── EXIT: Chandelier trailing stop ────────────────────────────
        # Check exit first — if price drops below chandelier long exit, close all
        if side == 1 and price < chand_long_now:
            context.close_long()
            self.position_scale = 0
            self.last_breakout_high = 0.0
            return

        # ── REDUCE: ADX weakening (20-25 range) ──────────────────────
        # If trend is losing steam but not dead, reduce exposure by half
        if side == 1 and self._ADX_WEAK_LOW <= adx_now < self.adx_threshold:
            half_lots = max(1, lots // 2)
            context.close_long(lots=half_lots)
            self.position_scale = max(0, self.position_scale - 1)
            return

        # ── ENTRY: Donchian breakout + strong ADX ─────────────────────
        if side == 0 and price > don_upper_now and adx_now > self.adx_threshold:
            lot_size = self._calc_lots(context, highs, chand_long_now)
            if lot_size > 0:
                context.buy(lot_size)
                self.position_scale = 1
                self.last_breakout_high = don_upper_now
            return

        # ── ADD: New higher Donchian breakout + ADX still strong ──────
        # Scale into position on successive higher breakouts (max scale 3)
        if (
            side == 1
            and self.position_scale < self._MAX_SCALE
            and don_upper_now > self.last_breakout_high
            and price > don_upper_now
            and adx_now > self.adx_threshold
        ):
            lot_size = self._calc_lots(context, highs, chand_long_now)
            if lot_size > 0:
                context.buy(lot_size)
                self.position_scale += 1
                self.last_breakout_high = don_upper_now

    # ── Position sizing ───────────────────────────────────────────────
    def _calc_lots(self, context, highs, chand_long_now):
        """Size position so that 2% of equity is at risk per unit.

        Risk per lot = distance from highest high to chandelier long exit,
        which equals chand_mult * ATR (the risk embedded in the chandelier).
        Multiply by contract_multiplier to get dollar risk per lot.
        """
        # The chandelier long exit = highest_high - mult * ATR
        # So risk distance = highest_high - chand_long_now = mult * ATR
        highest_high = np.max(highs[-self.chand_period:])
        risk_distance = highest_high - chand_long_now  # = chand_mult * ATR

        if risk_distance <= 0:
            return 0

        # Contract multiplier (points-to-currency conversion, default 1)
        contract_mult = getattr(context, "contract_multiplier", 100)

        risk_per_lot = risk_distance * contract_mult
        risk_budget = context.equity * 0.02  # 2% risk per trade

        if risk_per_lot <= 0:
            return 0

        lot_size = int(risk_budget / risk_per_lot)
        return max(1, min(lot_size, 50))
