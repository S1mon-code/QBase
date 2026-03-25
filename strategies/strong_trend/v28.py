"""
Strong Trend Strategy v28 — Donchian + Choppiness Index + A/D Line
===================================================================
Captures large trending moves (>100%, 6+ months) using:
  1. Donchian Channel  — breakout detection & trend direction
  2. Choppiness Index  — trending vs choppy filter (low = trending)
  3. A/D Line          — accumulation/distribution confirmation

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v28.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v28.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.donchian import donchian
from indicators.momentum.chop_zone import choppiness_index
from indicators.volume.ad_line import ad_line
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV28(TimeSeriesStrategy):
    """Donchian + Choppiness Index + A/D Line — long-only strong trend strategy."""

    name = "strong_trend_v28"
    warmup = 80
    freq = "daily"

    # ----- Tunable parameters (5 total) -----
    don_period: int = 40           # Donchian channel lookback
    chop_period: int = 14          # Choppiness Index period
    chop_threshold: float = 50.0   # Max choppiness to enter (lower = more trending)
    ad_lookback: int = 10          # Bars to compare A/D direction
    atr_trail_mult: float = 4.5   # Trailing stop ATR multiplier

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        lookback = max(self.warmup, self.don_period + 10, self.chop_period + 10,
                       self.ad_lookback + 10)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Compute indicators -----
        don_upper, don_lower, don_middle = donchian(highs, lows, self.don_period)
        chop = choppiness_index(highs, lows, closes, self.chop_period)
        ad = ad_line(highs, lows, closes, volumes)
        atr_vals = atr(highs, lows, closes, period=14)

        # Current values
        cur_don_upper = don_upper[-1]
        cur_don_middle = don_middle[-1]
        cur_chop = chop[-1]
        cur_atr = atr_vals[-1]
        price = context.current_bar.close_raw
        cur_close = closes[-1]

        # A/D Line direction: compare current vs N bars ago
        ad_rising = (
            len(ad) >= self.ad_lookback + 1
            and not np.isnan(ad[-1])
            and not np.isnan(ad[-(self.ad_lookback + 1)])
            and ad[-1] > ad[-(self.ad_lookback + 1)]
        )
        ad_falling = (
            len(ad) >= self.ad_lookback + 1
            and not np.isnan(ad[-1])
            and not np.isnan(ad[-(self.ad_lookback + 1)])
            and ad[-1] < ad[-(self.ad_lookback + 1)]
        )

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_don_upper) or np.isnan(cur_don_middle)
                or np.isnan(cur_chop) or np.isnan(cur_atr)):
            return

        # Derived signals
        above_upper = cur_close > cur_don_upper
        above_middle = cur_close > cur_don_middle
        trending = cur_chop < self.chop_threshold
        very_trending = cur_chop < 40.0
        choppy = cur_chop > 70.0

        side, lots = context.position

        # Update trailing stop (only ratchet upward)
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — close < Donchian middle OR Choppiness > 70
        #        OR trailing stop hit
        # ==================================================================
        if lots > 0:
            below_middle = cur_close < cur_don_middle
            if below_middle or choppy or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — A/D falling but Donchian still bullish → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if ad_falling and above_middle:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD POSITION — Choppiness < 40 (very trending) + A/D still rising
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS

            if profit_ok and bar_gap_ok and very_trending and ad_rising:
                base_lots = self._calc_lots(context, cur_atr)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, close > Donchian upper + Choppiness < threshold
        #         + A/D line rising
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            if above_upper and trending and ad_rising:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, atr_now: float) -> int:
        """Size position based on ATR-based trailing stop distance."""
        risk_per_trade = context.equity * 0.02
        risk_per_lot = self.atr_trail_mult * atr_now * self.contract_multiplier

        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
