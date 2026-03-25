"""
Strong Trend Strategy v46 — Supertrend Only (BASELINE)
=======================================================
The simplest possible strong trend strategy using a single indicator.
If other strategies can't beat this, they're not worth the complexity.

  1. Supertrend — trend direction, entry, and exit

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v46.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v46.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV46(TimeSeriesStrategy):
    """Supertrend only — long-only baseline strong trend strategy."""

    name = "strong_trend_v46"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    st_period: int = 10           # Supertrend ATR lookback
    st_mult: float = 3.0          # Supertrend ATR multiplier
    atr_period: int = 14          # ATR lookback (for trailing stop & sizing)
    atr_trail_mult: float = 5.0   # Trailing stop ATR multiplier
    new_high_lookback: int = 20   # Bars to check for new high (add signal)

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.highest_since_entry = 0.0
        self.bars_since_last_scale = 999  # large initial value

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        lookback = max(self.warmup, self.st_period + 10, self.new_high_lookback + 5)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Compute indicators -----
        st_line, st_dir = supertrend(highs, lows, closes, self.st_period, self.st_mult)
        atr_vals = atr(highs, lows, closes, period=self.atr_period)

        # Current values
        cur_dir = st_dir[-1]
        prev_dir = st_dir[-2] if len(st_dir) >= 2 else np.nan
        cur_st_line = st_line[-1]
        cur_atr = atr_vals[-1]
        price = context.current_bar.close_raw

        # Guard: skip if indicators aren't ready
        if np.isnan(cur_dir) or np.isnan(cur_st_line) or np.isnan(cur_atr):
            return

        side, lots = context.position

        # Track highest price since entry
        if lots > 0:
            if price > self.highest_since_entry:
                self.highest_since_entry = price

        # Update trailing stop (based on highest price since entry)
        if lots > 0 and cur_atr > 0:
            new_trail = self.highest_since_entry - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — Supertrend flips bearish OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if cur_dir == -1 or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                self.highest_since_entry = 0.0
                return

        # ==================================================================
        # ADD — New 20-bar high + Supertrend still bullish + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS

            # Check for new high: current close is highest in last N bars
            recent_highs = closes[-self.new_high_lookback:]
            new_high = price >= np.max(recent_highs)
            st_bullish = cur_dir == 1

            if profit_ok and bar_gap_ok and new_high and st_bullish:
                base_lots = self._calc_lots(context, price, cur_st_line)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, Supertrend flips from bearish to bullish
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            bullish_flip = (not np.isnan(prev_dir)
                            and prev_dir == -1
                            and cur_dir == 1)

            if bullish_flip:
                entry_lots = self._calc_lots(context, price, cur_st_line)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.highest_since_entry = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, st_line: float) -> int:
        """Size position based on distance to Supertrend line."""
        risk_per_trade = context.equity * 0.02
        distance = abs(price - st_line)

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
