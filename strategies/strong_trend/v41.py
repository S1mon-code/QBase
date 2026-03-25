"""
Strong Trend Strategy v41 — Multi-Timeframe: 4h Supertrend + 30min RSI
=======================================================================
Runs on 30min bars, aggregates to 4h for trend direction (Supertrend),
uses 30min RSI for entry timing (buy oversold dips in uptrend).

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v41.py --symbols AG --freq 30min --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v41.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.supertrend import supertrend
from indicators.momentum.rsi import rsi
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV41(TimeSeriesStrategy):
    """Multi-TF: 4h Supertrend direction + 30min RSI entry — long-only."""

    name = "strong_trend_v41"
    warmup = 500
    freq = "30min"

    # ----- Tunable parameters (<=5) -----
    st_period: int = 10          # Supertrend ATR lookback (on 4h bars)
    st_mult: float = 3.0        # Supertrend ATR multiplier
    rsi_period: int = 14         # RSI lookback (on 30min bars)
    rsi_entry: float = 35.0     # RSI oversold threshold for entry
    atr_trail_mult: float = 4.0 # Trailing stop ATR multiplier (30min ATR)

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value

    # ------------------------------------------------------------------
    # 4h bar aggregation from 30min bars
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_4h(closes_30m, highs_30m, lows_30m):
        """Aggregate 30min bars into 4h bars (every 8 bars).

        Returns (closes_4h, highs_4h, lows_4h) numpy arrays.
        """
        n = len(closes_30m)
        n_4h = n // 8

        if n_4h < 2:
            return None, None, None

        # Trim to exact multiple of 8
        trim = n_4h * 8

        groups_c = closes_30m[:trim].reshape(n_4h, 8)
        groups_h = highs_30m[:trim].reshape(n_4h, 8)
        groups_l = lows_30m[:trim].reshape(n_4h, 8)

        closes_4h = groups_c[:, -1]       # last close of each group
        highs_4h = np.max(groups_h, axis=1)  # highest high
        lows_4h = np.min(groups_l, axis=1)   # lowest low

        return closes_4h, highs_4h, lows_4h

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every 30min bar."""
        # Need enough 30min bars for 4h aggregation + indicator warmup
        # 8 bars per 4h candle * (st_period + 20) ~ 240 bars minimum
        lookback = max(self.warmup, (self.st_period + 30) * 8, self.rsi_period * 3)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Aggregate to 4h bars -----
        closes_4h, highs_4h, lows_4h = self._aggregate_4h(closes, highs, lows)
        if closes_4h is None or len(closes_4h) < self.st_period + 5:
            return

        # ----- Compute 4h Supertrend (large TF: direction) -----
        st_line_4h, st_dir_4h = supertrend(
            highs_4h, lows_4h, closes_4h, self.st_period, self.st_mult
        )
        cur_st_dir = st_dir_4h[-1]

        if np.isnan(cur_st_dir):
            return

        # ----- Compute 30min RSI (small TF: entry timing) -----
        rsi_vals = rsi(closes, self.rsi_period)
        cur_rsi = rsi_vals[-1]

        if np.isnan(cur_rsi):
            return

        # ----- Compute 30min ATR (for trailing stop + profit check) -----
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
        # EXIT — 4h Supertrend flips bearish OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if cur_st_dir == -1 or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — RSI > 80 (overbought) → take partial profit, close half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_rsi > 80.0:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — 4h Supertrend still bullish + RSI < 30 (deeper pullback)
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            st_bullish = cur_st_dir == 1
            deep_pullback = cur_rsi < 30.0

            if profit_ok and bar_gap_ok and st_bullish and deep_pullback:
                base_lots = self._calc_lots(context, price, cur_atr)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, 4h Supertrend bullish + 30min RSI < 35 (oversold dip)
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            st_bullish = cur_st_dir == 1
            rsi_oversold = cur_rsi < self.rsi_entry

            if st_bullish and rsi_oversold:
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
