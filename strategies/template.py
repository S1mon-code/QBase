"""
QBase Strategy Template (Pre-computation Architecture)
======================================================
Copy this file to strategies/<category>/v<n>.py and modify.

Uses on_init_arrays for ~20x speedup over traditional on_bar computation.

Usage:
    ./run.sh strategies/<category>/v1.py --symbols <SYMBOL> --freq daily --start 2022

Naming convention:
    strategies/strong_trend/v1.py   — 强趋势策略 v1
    strategies/mean_reversion/v1.py — 震荡/均值回归策略 v1
    strategies/all_time/ag/v1.py    — 白银全时间策略 v1
"""
import sys
from pathlib import Path

# Ensure QBase root is in path for indicator imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

# Import indicators as needed
from indicators.trend.adx import adx
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr


class TemplateStrategy(TimeSeriesStrategy):
    name = "template_strategy"
    warmup = 60  # Bars needed before trading starts
    freq = "daily"  # or "5min", "10min", "30min", "1h", "4h"

    # ----- Tunable parameters (<=5) -----
    adx_period: int = 14
    roc_period: int = 20
    atr_trail_mult: float = 3.0

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        # Declare pre-computed indicator arrays
        self._adx = None
        self._roc = None
        self._atr = None

    def on_init(self, context):
        """Initialize state variables (trading state, NOT indicators)."""
        self.entry_price = 0.0

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays. Called once."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._adx = adx(highs, lows, closes, period=self.adx_period)
        self._roc = rate_of_change(closes, self.roc_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        """Called on every bar — pure lookup + trading logic."""
        i = context.bar_index

        # O(1) indicator lookup
        adx_val = self._adx[i]
        roc_val = self._roc[i]
        atr_val = self._atr[i]

        # Skip if indicators not ready
        if np.isnan(adx_val) or np.isnan(roc_val) or np.isnan(atr_val):
            return

        price = context.close_raw
        side, lots = context.position

        # === Entry Logic ===
        if lots == 0:
            if adx_val > 25 and roc_val > 0:
                # Strong uptrend → buy
                lot_size = self._calc_lots(context, price, atr_val)
                context.buy(lot_size)
                self.entry_price = price
            elif adx_val > 25 and roc_val < 0:
                # Strong downtrend → sell
                lot_size = self._calc_lots(context, price, atr_val)
                context.sell(lot_size)
                self.entry_price = price

        # === Exit Logic ===
        elif side == 1:
            # Exit long: trend weakens or stop loss
            if adx_val < 20 or price < self.entry_price - 2 * atr_val:
                context.close_long()
        elif side == -1:
            # Exit short: trend weakens or stop loss
            if adx_val < 20 or price > self.entry_price + 2 * atr_val:
                context.close_short()

    def _calc_lots(self, context, price, atr_val):
        """Position sizing based on ATR risk."""
        risk_per_trade = context.equity * 0.02  # Risk 2% per trade
        risk_per_lot = atr_val * 2 * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1
        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
