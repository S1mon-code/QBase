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

    def __init__(self):
        super().__init__()
        # Small TF indicators (30min)
        self._rsi = None
        self._atr = None
        # Large TF indicators (4h, pre-mapped to 30min indices)
        self._st_line_4h = None
        self._st_dir_4h = None
        self._4h_map = None  # 30min index → 4h index mapping

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators once. Aggregate 30min → 4h."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        n = len(closes)

        # Small TF indicators (30min)
        self._rsi = rsi(closes, self.rsi_period)
        self._atr = atr(highs, lows, closes, period=14)

        # Aggregate to 4h (step = 8: 8 × 30min = 4h)
        step = 8
        n_4h = n // step
        trim = n_4h * step
        closes_4h = closes[:trim].reshape(n_4h, step)[:, -1]
        highs_4h = highs[:trim].reshape(n_4h, step).max(axis=1)
        lows_4h = lows[:trim].reshape(n_4h, step).min(axis=1)

        # Large TF indicators (4h)
        self._st_line_4h, self._st_dir_4h = supertrend(
            highs_4h, lows_4h, closes_4h, self.st_period, self.st_mult
        )

        # Index mapping: for 30min bar i, the latest COMPLETED 4h bar
        # This avoids look-ahead bias
        self._4h_map = np.maximum(0, (np.arange(n) + 1) // step - 1)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every 30min bar."""
        i = context.bar_index
        j = self._4h_map[i]  # corresponding 4h bar

        self.bars_since_last_scale += 1

        # ----- Lookup pre-computed values -----
        cur_st_dir = self._st_dir_4h[j]
        cur_rsi = self._rsi[i]
        cur_atr = self._atr[i]

        # NaN guard
        if np.isnan(cur_st_dir) or np.isnan(cur_rsi) or np.isnan(cur_atr) or cur_atr <= 0:
            return

        price = context.close_raw
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
