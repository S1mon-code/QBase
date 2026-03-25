"""
Strong Trend Strategy v48 — R-squared + Donchian
=================================================
Captures large trending moves using only two indicators:
  1. R-squared   — trend quality (goodness of linear fit)
  2. Donchian    — channel breakout (highest high / lowest low)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v48.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.linear_regression import r_squared
from indicators.trend.donchian import donchian
from indicators.volatility.atr import atr


SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV48(TimeSeriesStrategy):
    """R-squared + Donchian — long-only strong trend strategy."""

    name = "strong_trend_v48"
    warmup = 60
    freq = "daily"

    # Tunable parameters (<=5)
    rsq_period: int = 40
    don_period: int = 40
    atr_period: int = 14
    atr_trail_mult: float = 4.5
    rsq_threshold: float = 0.5

    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        self._rsq = None
        self._don_upper = None
        self._don_lower = None
        self._atr = None

    def on_init(self, context):
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999
        self.prev_don_upper = 0.0

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators once."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._rsq = r_squared(closes, self.rsq_period)
        self._don_upper, self._don_lower, _ = donchian(highs, lows, self.don_period)
        self._atr = atr(highs, lows, closes, period=self.atr_period)

    def on_bar(self, context):
        i = context.bar_index
        self.bars_since_last_scale += 1

        # Lookup pre-computed indicators
        cur_rsq = self._rsq[i]
        cur_don_upper = self._don_upper[i]
        cur_don_lower = self._don_lower[i]
        cur_atr = self._atr[i]
        price = context.close_raw

        if np.isnan(cur_rsq) or np.isnan(cur_don_upper) or np.isnan(cur_don_lower) or np.isnan(cur_atr):
            return

        side, lots = context.position

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # === EXIT — close < Donchian lower OR R² < 0.2 OR trailing stop ===
        if lots > 0:
            if price < cur_don_lower or cur_rsq < 0.2 or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                self.prev_don_upper = 0.0
                return

        # === REDUCE — R² drops below 0.3 → close half ===
        if lots > 0 and self.position_scale > 0:
            if cur_rsq < 0.3:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # === ADD — R² > 0.6 + new Donchian breakout (higher high) ===
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            rsq_strong = cur_rsq > 0.6
            new_breakout = cur_don_upper > self.prev_don_upper and price > cur_don_upper

            if profit_ok and bar_gap_ok and rsq_strong and new_breakout:
                base_lots = self._calc_lots(context, price, cur_atr)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                    self.prev_don_upper = cur_don_upper
                return

        # === ENTRY — flat, R² > threshold + close > Donchian upper ===
        if lots == 0 and self.position_scale == 0:
            if cur_rsq > self.rsq_threshold and price > cur_don_upper:
                entry_lots = self._calc_lots(context, price, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0
                    self.prev_don_upper = cur_don_upper

    def _calc_lots(self, context, price: float, cur_atr: float) -> int:
        """Size position based on ATR risk distance."""
        risk_per_trade = context.equity * 0.02
        distance = self.atr_trail_mult * cur_atr

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
