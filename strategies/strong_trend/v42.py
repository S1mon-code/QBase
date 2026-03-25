"""
Strong Trend Strategy v42 — Multi-Timeframe: Daily ADX + 1h EMA Cross
======================================================================
Runs on 60min bars, aggregates to daily for trend strength (ADX),
uses 1h EMA fast/slow crossover for entry timing.

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v42.py --symbols AG --freq 60min --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v42.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.adx import adx
from indicators.trend.ema import ema
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV42(TimeSeriesStrategy):
    """Multi-TF: daily ADX trend filter + 1h EMA cross entry — long-only."""

    name = "strong_trend_v42"
    warmup = 300
    freq = "60min"

    # ----- Tunable parameters (<=5) -----
    adx_period: int = 14          # ADX lookback (on daily bars)
    adx_threshold: float = 25.0   # Min ADX to enter
    ema_fast: int = 20            # Fast EMA period (on 1h bars)
    ema_slow: int = 50            # Slow EMA period (on 1h bars)
    atr_trail_mult: float = 4.0   # Trailing stop ATR multiplier (1h ATR)

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        # Small TF indicators (1h)
        self._ema_fast = None
        self._ema_slow = None
        self._atr = None
        # Large TF indicators (daily, pre-mapped to 1h indices)
        self._adx_daily = None
        self._daily_map = None  # 1h index → daily index mapping

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators once. Aggregate 60min → daily."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        n = len(closes)

        # Small TF indicators (1h)
        self._ema_fast = ema(closes, self.ema_fast)
        self._ema_slow = ema(closes, self.ema_slow)
        self._atr = atr(highs, lows, closes, period=14)

        # Aggregate to daily (step = 4: 4 × 60min ≈ 1 session)
        step = 4
        n_daily = n // step
        trim = n_daily * step
        closes_d = closes[:trim].reshape(n_daily, step)[:, -1]
        highs_d = highs[:trim].reshape(n_daily, step).max(axis=1)
        lows_d = lows[:trim].reshape(n_daily, step).min(axis=1)

        # Large TF indicators (daily)
        self._adx_daily = adx(highs_d, lows_d, closes_d, self.adx_period)

        # Index mapping: for 1h bar i, the latest COMPLETED daily bar
        self._daily_map = np.maximum(0, (np.arange(n) + 1) // step - 1)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every 1h bar."""
        i = context.bar_index
        j = self._daily_map[i]  # corresponding daily bar

        self.bars_since_last_scale += 1

        # ----- Lookup pre-computed values -----
        cur_adx = self._adx_daily[j]
        cur_ema_fast = self._ema_fast[i]
        cur_ema_slow = self._ema_slow[i]
        cur_atr = self._atr[i]

        # NaN guard
        if np.isnan(cur_adx) or np.isnan(cur_ema_fast) or np.isnan(cur_ema_slow):
            return
        if np.isnan(cur_atr) or cur_atr <= 0:
            return

        # Previous bar EMA values (for cross detection)
        if i < 1:
            return
        prev_ema_fast = self._ema_fast[i - 1]
        prev_ema_slow = self._ema_slow[i - 1]
        if np.isnan(prev_ema_fast) or np.isnan(prev_ema_slow):
            return

        # EMA cross detection
        ema_golden_cross = (prev_ema_fast <= prev_ema_slow) and (cur_ema_fast > cur_ema_slow)
        ema_fast_below = cur_ema_fast < cur_ema_slow

        # EMA spread (for add condition — widening means trend accelerating)
        cur_spread = cur_ema_fast - cur_ema_slow
        prev_spread = prev_ema_fast - prev_ema_slow
        spread_widening = cur_spread > prev_spread and cur_spread > 0

        price = context.close_raw
        side, lots = context.position

        # ----- Update trailing stop -----
        if lots > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — daily ADX < 15 (no trend) OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if cur_adx < 15.0 or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — 1h EMA fast < slow but daily ADX still > 20 → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if ema_fast_below and cur_adx > 20.0:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — daily ADX > 35 + EMA spread widening + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            adx_strong = cur_adx > 35.0

            if profit_ok and bar_gap_ok and adx_strong and spread_widening:
                base_lots = self._calc_lots(context, price, cur_atr)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, daily ADX > threshold + 1h EMA golden cross
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            adx_ok = cur_adx > self.adx_threshold
            cross_ok = ema_golden_cross

            if adx_ok and cross_ok:
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
        """Size position based on ATR-based stop distance."""
        risk_per_trade = context.equity * 0.02
        distance = self.atr_trail_mult * cur_atr

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
