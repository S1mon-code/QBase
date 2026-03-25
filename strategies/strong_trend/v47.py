"""
Strong Trend Strategy v47 — ADX + Chandelier Exit
===================================================
Captures large trending moves using only two indicators:
  1. ADX           — trend strength filter
  2. Chandelier Exit — adaptive trailing stop (highest high - ATR*mult)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v47.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.adx import adx
from indicators.volatility.chandelier_exit import chandelier_exit
from indicators.volatility.atr import atr


SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV47(TimeSeriesStrategy):
    """ADX + Chandelier Exit — long-only strong trend strategy."""

    name = "strong_trend_v47"
    warmup = 60
    freq = "daily"

    # Tunable parameters (<=5)
    adx_period: int = 14
    adx_threshold: float = 25.0
    chand_period: int = 22
    chand_mult: float = 3.0
    atr_trail_mult: float = 4.5

    contract_multiplier: float = 100.0

    def on_init(self, context):
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999

    def on_bar(self, context):
        lookback = max(self.warmup, self.adx_period * 3, self.chand_period + 10)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # Compute indicators
        adx_vals = adx(highs, lows, closes, self.adx_period)
        chand_long, _ = chandelier_exit(highs, lows, closes, self.chand_period, self.chand_mult)
        atr_vals = atr(highs, lows, closes, period=14)

        cur_adx = adx_vals[-1]
        cur_chand = chand_long[-1]
        cur_atr = atr_vals[-1]
        price = context.current_bar.close_raw

        if np.isnan(cur_adx) or np.isnan(cur_chand) or np.isnan(cur_atr):
            return

        side, lots = context.position

        # Update trailing stop (backup)
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # === EXIT — price < Chandelier long exit OR backup trailing stop ===
        if lots > 0:
            if price < cur_chand or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # === REDUCE — ADX drops below 20 → close half ===
        if lots > 0 and self.position_scale > 0:
            if cur_adx < 20.0:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # === ADD — ADX very strong + new 20-bar high ===
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            adx_strong = cur_adx > 35.0
            new_high = price >= np.max(highs[-20:])

            if profit_ok and bar_gap_ok and adx_strong and new_high:
                base_lots = self._calc_lots(context, price, cur_chand)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # === ENTRY — flat, ADX > threshold + price above Chandelier long exit ===
        if lots == 0 and self.position_scale == 0:
            if cur_adx > self.adx_threshold and price > cur_chand:
                entry_lots = self._calc_lots(context, price, cur_chand)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    def _calc_lots(self, context, price: float, chand_line: float) -> int:
        """Size position based on distance to Chandelier exit."""
        risk_per_trade = context.equity * 0.02
        distance = abs(price - chand_line)

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
