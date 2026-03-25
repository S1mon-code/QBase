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

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value
        self.prev_cci = np.nan            # previous bar CCI for crossover detection
        self.prev_vi_spread = np.nan      # previous 4h VI+ - VI- spread

    # ------------------------------------------------------------------
    # Helper: aggregate 30min bars to 4h (every 8 bars)
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_to_4h(highs_30m, lows_30m, closes_30m):
        """Aggregate 30min OHLC arrays into 4h bars (8:1 ratio).

        Returns (highs_4h, lows_4h, closes_4h) trimmed to complete bars.
        """
        n = len(highs_30m)
        n_bars = n // 8
        if n_bars < 1:
            return np.array([]), np.array([]), np.array([])

        # Trim to exact multiple of 8 (use most recent bars)
        trim = n - n_bars * 8
        h = highs_30m[trim:].reshape(n_bars, 8)
        l = lows_30m[trim:].reshape(n_bars, 8)
        c = closes_30m[trim:].reshape(n_bars, 8)

        highs_4h = np.max(h, axis=1)
        lows_4h = np.min(l, axis=1)
        closes_4h = c[:, -1]  # last close in each 4h block

        return highs_4h, lows_4h, closes_4h

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        lookback = max(self.warmup, self.vortex_period * 8 + 50, self.cci_period + 10)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Compute 30min indicators -----
        cci_vals = cci(highs, lows, closes, self.cci_period)
        atr_vals = atr(highs, lows, closes, period=self.atr_period)

        # ----- Aggregate to 4h and compute Vortex -----
        h4h, l4h, c4h = self._aggregate_to_4h(highs, lows, closes)
        if len(h4h) < self.vortex_period + 2:
            return

        vi_plus, vi_minus = vortex(h4h, l4h, c4h, self.vortex_period)

        # Current values
        cur_cci = cci_vals[-1]
        prev_cci = cci_vals[-2] if len(cci_vals) >= 2 else np.nan
        cur_atr = atr_vals[-1]
        cur_vi_plus = vi_plus[-1]
        cur_vi_minus = vi_minus[-1]
        prev_vi_plus = vi_plus[-2] if len(vi_plus) >= 2 else np.nan
        prev_vi_minus = vi_minus[-2] if len(vi_minus) >= 2 else np.nan
        price = context.current_bar.close_raw

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
