"""
Strong Trend Strategy v37 — Fractal Levels + RWI + Force Index
===============================================================
Captures large trending moves (>100%, 6+ months) using:
  1. Fractal Levels — breakout detection via Williams fractals
  2. RWI           — trend strength (random walk index)
  3. Force Index   — volume-weighted momentum confirmation

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v37.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v37.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.fractal import fractal_levels
from indicators.trend.rwi import rwi
from indicators.volume.force_index import force_index
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV37(TimeSeriesStrategy):
    """Fractal Levels + RWI + Force Index — long-only strong trend strategy."""

    name = "strong_trend_v37"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    fractal_period: int = 2        # Fractal detection window (each side)
    rwi_period: int = 14           # Random Walk Index lookback
    fi_period: int = 13            # Force Index EMA period
    atr_period: int = 14           # ATR lookback (for sizing & trailing)
    atr_trail_mult: float = 4.5   # Trailing stop ATR multiplier

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value
        self.prev_fractal_high = 0.0      # track for add signal

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        lookback = max(self.warmup, self.rwi_period * 3, self.fi_period + 10)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Compute indicators -----
        frac_high, frac_low = fractal_levels(highs, lows, self.fractal_period)
        rwi_high, rwi_low = rwi(highs, lows, closes, self.rwi_period)
        fi_vals = force_index(closes, volumes, self.fi_period)
        atr_vals = atr(highs, lows, closes, period=self.atr_period)

        # Current values
        cur_frac_high = frac_high[-1]
        cur_frac_low = frac_low[-1]
        cur_rwi_high = rwi_high[-1]
        cur_fi = fi_vals[-1]
        cur_atr = atr_vals[-1]
        price = context.current_bar.close_raw

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_frac_high) or np.isnan(cur_frac_low)
                or np.isnan(cur_rwi_high) or np.isnan(cur_fi) or np.isnan(cur_atr)):
            return

        side, lots = context.position

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — close < last_fractal_low OR RWI high < 0.5 OR trailing stop
        # ==================================================================
        if lots > 0:
            if price < cur_frac_low or cur_rwi_high < 0.5 or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                self.prev_fractal_high = 0.0
                return

        # ==================================================================
        # REDUCE — Force Index < 0 but RWI still > 1.0 → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_fi < 0 and cur_rwi_high > 1.0:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — new higher fractal_high + RWI high > 1.5 + scale<3
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            new_higher_fractal = cur_frac_high > self.prev_fractal_high
            rwi_strong = cur_rwi_high > 1.5

            if profit_ok and bar_gap_ok and new_higher_fractal and rwi_strong:
                base_lots = self._calc_lots(context, price, cur_atr)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                    self.prev_fractal_high = cur_frac_high
                return

        # ==================================================================
        # ENTRY — flat, close > last_fractal_high + RWI > 1.0 + FI > 0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            breakout = price > cur_frac_high
            rwi_ok = cur_rwi_high > 1.0
            volume_ok = cur_fi > 0

            if breakout and rwi_ok and volume_ok:
                entry_lots = self._calc_lots(context, price, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0
                    self.prev_fractal_high = cur_frac_high

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, cur_atr: float) -> int:
        """Size position based on ATR distance."""
        risk_per_trade = context.equity * 0.02
        distance = self.atr_trail_mult * cur_atr

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
