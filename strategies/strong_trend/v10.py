"""
Strong Trend Strategy v10 — TEMA + ADX + Bollinger Width
=========================================================
Captures large trending moves using:
  1. TEMA           — trend direction & low-lag price filter
  2. ADX            — trend strength measurement
  3. Bollinger Width — volatility expansion detection

Entry logic: close > TEMA + TEMA rising + ADX strong + BB width expanding
= trending price + confirmed strength + volatility breakout = strong trend starting.

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v10.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v10.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.tema import tema
from indicators.trend.adx import adx
from indicators.volatility.bollinger import bollinger_width
from indicators.volatility.atr import atr


class StrongTrendV10(TimeSeriesStrategy):
    """TEMA + ADX + Bollinger Width — long-only strong trend strategy."""

    name = "strong_trend_v10"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (5) -----
    tema_period: int = 20           # TEMA lookback
    adx_period: int = 14            # ADX lookback
    adx_threshold: float = 25.0     # Min ADX to enter (trending)
    bb_period: int = 20             # Bollinger Width lookback
    atr_trail_mult: float = 3.0     # ATR multiplier for trailing stop & sizing

    # ----- Position sizing -----
    contract_multiplier: float = 100.0  # Default for most commodities

    def __init__(self):
        super().__init__()
        self._tema = None
        self._adx = None
        self._bb_width = None
        self._atr = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0         # 0 = flat, 1-3 = position tiers
        self.trail_stop = 0.0           # Trailing stop price

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._tema = tema(closes, self.tema_period)
        self._adx = adx(highs, lows, closes, self.adx_period)
        self._bb_width = bollinger_width(closes, self.bb_period)
        self._atr = atr(highs, lows, closes, self.adx_period)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index

        if i < 1:
            return

        # ----- Look up pre-computed indicators -----
        price = context.close_raw
        cur_tema = self._tema[i]
        prev_tema = self._tema[i - 1]
        cur_adx = self._adx[i]
        cur_atr = self._atr[i]

        # Guard: skip if indicators aren't ready
        if np.isnan(cur_tema) or np.isnan(cur_adx) or np.isnan(cur_atr):
            return

        # TEMA rising = current > previous
        tema_rising = cur_tema > prev_tema

        # Bollinger Width expanding = current > mean of last 20
        bb_start = max(0, i - 19)
        bb_recent = self._bb_width[bb_start:i + 1]
        bb_valid = bb_recent[~np.isnan(bb_recent)]
        if len(bb_valid) < 5:
            return
        cur_bb_width = self._bb_width[i]
        if np.isnan(cur_bb_width):
            return
        bb_width_expanding = cur_bb_width > np.mean(bb_valid)

        side, lots = context.position

        # Update trailing stop for active positions
        if lots > 0 and cur_atr > 0:
            new_stop = price - self.atr_trail_mult * cur_atr
            self.trail_stop = max(self.trail_stop, new_stop)

        # ==================================================================
        # EXIT — close below TEMA OR ADX dead OR trailing stop hit
        # ==================================================================
        if lots > 0:
            close_below_tema = price < cur_tema
            adx_dead = cur_adx < 15.0
            trail_stop_hit = price < self.trail_stop

            if close_below_tema or adx_dead or trail_stop_hit:
                context.close_long()
                self.position_scale = 0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — ADX drops below threshold but TEMA still rising
        # Close half the position (momentum weakening, trend intact)
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_adx < self.adx_threshold and tema_rising:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — ADX very strong + BB width still expanding + room to scale
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            if cur_adx > 35.0 and bb_width_expanding:
                add_lots = self._calc_lots(context, cur_atr)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                return

        # ==================================================================
        # ENTRY — flat, close > TEMA, TEMA rising, ADX strong, BB expanding
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            above_tema = price > cur_tema
            adx_strong = cur_adx > self.adx_threshold

            if above_tema and tema_rising and adx_strong and bb_width_expanding:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    # Initialize trailing stop
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
        stop_distance = self.atr_trail_mult * cur_atr

        if stop_distance <= 0:
            return 1  # Fallback: minimum lot

        risk_per_lot = stop_distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))  # Clamp to [1, 50]
