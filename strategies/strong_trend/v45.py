"""
Strong Trend Strategy v45 — Multi-Timeframe Vortex + CCI
=========================================================
Captures large trending moves using:
  1. 4h Vortex  — trend direction (aggregated from 30min bars)
  2. 30min CCI  — entry timing (oversold recovery)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v45.py --symbols AG --freq 30min --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v45.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.vortex import vortex
from indicators.momentum.cci import cci
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV45(TimeSeriesStrategy):
    """Multi-TF Vortex + CCI — long-only strong trend strategy."""

    name = "strong_trend_v45"
    warmup = 500
    freq = "30min"

    # ----- Tunable parameters (<=5) -----
    vortex_period: int = 14       # Vortex lookback (applied on 4h bars)
    cci_period: int = 20          # CCI lookback (applied on 30min bars)
    cci_entry: float = -100.0     # CCI threshold for entry (recovery from oversold)
    atr_period: int = 14          # ATR lookback
    atr_trail_mult: float = 4.0   # Trailing stop ATR multiplier

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        # Small TF indicators (30min)
        self._cci = None
        self._atr = None
        # Large TF indicators (4h, pre-mapped to 30min indices)
        self._vi_plus_4h = None
        self._vi_minus_4h = None
        self._4h_map = None  # 30min index → 4h index mapping

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value
        self.prev_cci = np.nan            # previous bar CCI for crossover detection
        self.prev_vi_spread = np.nan      # previous 4h VI+ - VI- spread

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators once. Aggregate 30min → 4h."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        n = len(closes)

        # Small TF indicators (30min)
        self._cci = cci(highs, lows, closes, self.cci_period)
        self._atr = atr(highs, lows, closes, period=self.atr_period)

        # Aggregate to 4h (step = 8: 8 × 30min = 4h)
        step = 8
        n_4h = n // step
        trim = n_4h * step
        closes_4h = closes[:trim].reshape(n_4h, step)[:, -1]
        highs_4h = highs[:trim].reshape(n_4h, step).max(axis=1)
        lows_4h = lows[:trim].reshape(n_4h, step).min(axis=1)

        # Large TF indicators (4h Vortex)
        self._vi_plus_4h, self._vi_minus_4h = vortex(
            highs_4h, lows_4h, closes_4h, self.vortex_period
        )

        # Index mapping: for 30min bar i, the latest COMPLETED 4h bar
        self._4h_map = np.maximum(0, (np.arange(n) + 1) // step - 1)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index
        j = self._4h_map[i]  # corresponding 4h bar

        self.bars_since_last_scale += 1

        # ----- Lookup pre-computed values -----
        cur_cci = self._cci[i]
        cur_atr = self._atr[i]
        cur_vi_plus = self._vi_plus_4h[j]
        cur_vi_minus = self._vi_minus_4h[j]

        # Previous values for crossover / spread detection
        prev_cci = self._cci[i - 1] if i >= 1 else np.nan
        prev_vi_plus = self._vi_plus_4h[j - 1] if j >= 1 else np.nan
        prev_vi_minus = self._vi_minus_4h[j - 1] if j >= 1 else np.nan
        price = context.close_raw

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_cci) or np.isnan(prev_cci)
                or np.isnan(cur_atr) or np.isnan(cur_vi_plus)
                or np.isnan(cur_vi_minus)):
            return

        # Derived signals
        vortex_bullish = cur_vi_plus > cur_vi_minus
        cci_cross_above_entry = prev_cci < self.cci_entry and cur_cci >= self.cci_entry
        cur_vi_spread = cur_vi_plus - cur_vi_minus
        prev_vi_spread = prev_vi_plus - prev_vi_minus if not np.isnan(prev_vi_plus) else 0.0
        vi_spread_widening = cur_vi_spread > prev_vi_spread and cur_vi_spread > 0

        side, lots = context.position

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — 4h VI- > VI+ (bearish) OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if cur_vi_minus > cur_vi_plus or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — 30min CCI > 200 (extremely overbought) → close half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_cci > 200.0:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — 4h VI spread widening + 30min CCI > 100 + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS

            if profit_ok and bar_gap_ok and vi_spread_widening and cur_cci > 100.0:
                base_lots = self._calc_lots(context, price, cur_atr)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, 4h VI+ > VI- + 30min CCI crosses above -100
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            if vortex_bullish and cci_cross_above_entry:
                entry_lots = self._calc_lots(context, price, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

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
