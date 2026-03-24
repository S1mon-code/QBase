"""
QBase Strategy Template
=======================
Copy this file to strategies/<symbol>/<type>_v<n>.py and modify.

Usage:
    ./run.sh strategies/<symbol>/<type>_v1.py --symbols <SYMBOL> --freq daily --start 2022

Naming convention:
    trend_v1.py          — 趋势策略 v1
    mean_reversion_v1.py — 震荡/均值回归策略 v1
    breakout_v1.py       — 突破策略 v1
"""
import sys
from pathlib import Path

# Ensure QBase root is in path for indicator imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

# Import indicators as needed
from indicators.trend.adx import adx
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr


class TemplateStrategy(TimeSeriesStrategy):
    name = "template_strategy"
    warmup = 60  # Bars needed before trading starts

    def on_init(self, context):
        """Initialize state variables."""
        self.entry_price = 0.0

    def on_bar(self, context):
        """Called on every bar — core strategy logic."""
        # Get price arrays
        closes = context.get_close_array(60)
        highs = context.get_high_array(60)
        lows = context.get_low_array(60)

        price = context.current_bar.close_raw
        side, lots = context.position

        # Calculate indicators
        adx_val = adx(highs, lows, closes, period=14)[-1]
        roc_val = rate_of_change(closes, 20)[-1]
        atr_val = atr(highs, lows, closes, period=14)[-1]

        # Skip if indicators not ready
        if np.isnan(adx_val) or np.isnan(roc_val) or np.isnan(atr_val):
            return

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
        risk_per_lot = atr_val * 2 * 100  # 2x ATR * multiplier (adjust per symbol)
        if risk_per_lot <= 0:
            return 1
        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
