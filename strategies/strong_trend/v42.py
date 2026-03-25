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

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value

    # ------------------------------------------------------------------
    # Daily bar aggregation from 60min bars
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_daily(closes_1h, highs_1h, lows_1h, agg_size=4):
        """Aggregate 1h bars into daily bars (every agg_size bars).

        Uses ~4 bars per day as approximation (Chinese futures have ~4h
        of active trading per session). Returns (closes_d, highs_d, lows_d).
        """
        n = len(closes_1h)
        n_daily = n // agg_size

        if n_daily < 2:
            return None, None, None

        trim = n_daily * agg_size

        groups_c = closes_1h[:trim].reshape(n_daily, agg_size)
        groups_h = highs_1h[:trim].reshape(n_daily, agg_size)
        groups_l = lows_1h[:trim].reshape(n_daily, agg_size)

        closes_d = groups_c[:, -1]          # last close of each group
        highs_d = np.max(groups_h, axis=1)  # highest high
        lows_d = np.min(groups_l, axis=1)   # lowest low

        return closes_d, highs_d, lows_d

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every 1h bar."""
        lookback = max(self.warmup, (self.adx_period * 3) * 4 + 20, self.ema_slow + 20)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Aggregate to daily bars -----
        closes_d, highs_d, lows_d = self._aggregate_daily(closes, highs, lows)
        if closes_d is None or len(closes_d) < self.adx_period * 3:
            return

        # ----- Compute daily ADX (large TF: trend strength) -----
        adx_vals = adx(highs_d, lows_d, closes_d, self.adx_period)
        cur_adx = adx_vals[-1]

        if np.isnan(cur_adx):
            return

        # ----- Compute 1h EMA fast/slow (small TF: entry timing) -----
        ema_fast_vals = ema(closes, self.ema_fast)
        ema_slow_vals = ema(closes, self.ema_slow)
        cur_ema_fast = ema_fast_vals[-1]
        cur_ema_slow = ema_slow_vals[-1]
        prev_ema_fast = ema_fast_vals[-2]
        prev_ema_slow = ema_slow_vals[-2]

        if np.isnan(cur_ema_fast) or np.isnan(cur_ema_slow):
            return

        # EMA cross detection
        ema_golden_cross = (prev_ema_fast <= prev_ema_slow) and (cur_ema_fast > cur_ema_slow)
        ema_fast_above = cur_ema_fast > cur_ema_slow
        ema_fast_below = cur_ema_fast < cur_ema_slow

        # EMA spread (for add condition — widening means trend accelerating)
        cur_spread = cur_ema_fast - cur_ema_slow
        prev_spread = prev_ema_fast - prev_ema_slow
        spread_widening = cur_spread > prev_spread and cur_spread > 0

        # ----- Compute 1h ATR (for trailing stop + profit check) -----
        atr_vals = atr(highs, lows, closes, period=14)
        cur_atr = atr_vals[-1]

        if np.isnan(cur_atr) or cur_atr <= 0:
            return

        price = context.current_bar.close_raw
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
