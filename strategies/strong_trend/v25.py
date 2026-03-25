"""
Strong Trend Strategy v25 — Linear Regression Slope + Vortex + Volume Spike
=============================================================================
Captures large trending moves using:
  1. Linear Regression Slope — trend direction & acceleration
  2. Vortex Indicator        — bullish/bearish trend confirmation
  3. Volume Spike            — conviction filter

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v25.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v25.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.linear_regression import linear_regression_slope
from indicators.trend.vortex import vortex
from indicators.volume.volume_spike import volume_spike, volume_dry_up
from indicators.volatility.atr import atr


# Pyramid scale factors: 100% -> 50% -> 25%
SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrongTrendV25(TimeSeriesStrategy):
    """LinReg Slope + Vortex + Volume Spike — long-only strong trend strategy."""

    name = "strong_trend_v25"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    slope_period: int = 20       # Linear regression slope lookback
    vortex_period: int = 14      # Vortex indicator period
    vol_period: int = 20         # Volume spike / dry-up lookback
    vol_threshold: float = 2.0   # Volume spike detection threshold
    atr_trail_mult: float = 4.5  # ATR multiplier for trailing stop

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        self._slope = None
        self._vi_plus = None
        self._vi_minus = None
        self._vol_spikes = None
        self._vol_dry = None
        self._atr = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0          # 0 = flat, 1-3 = position tiers
        self.entry_price = 0.0           # Average entry price
        self.trail_stop = 0.0            # Trailing stop level
        self.bars_since_last_scale = 0   # Min gap between adds
        self.prev_slope = np.nan         # Previous bar slope for acceleration
        self.prev_vi_spread = np.nan     # Previous VI+ - VI- spread

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._slope = linear_regression_slope(closes, self.slope_period)
        self._vi_plus, self._vi_minus = vortex(highs, lows, closes, self.vortex_period)
        self._vol_spikes = volume_spike(volumes, period=self.vol_period,
                                        threshold=self.vol_threshold)
        self._vol_dry = volume_dry_up(volumes, period=self.vol_period, threshold=0.5)
        self._atr = atr(highs, lows, closes, period=14)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index

        if i < 2:
            return

        # ----- Look up pre-computed indicators -----
        cur_slope = self._slope[i]
        cur_vi_plus = self._vi_plus[i]
        cur_vi_minus = self._vi_minus[i]
        cur_vi_spread = cur_vi_plus - cur_vi_minus
        cur_atr = self._atr[i]
        price = context.close_raw

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_slope) or np.isnan(cur_vi_plus)
                or np.isnan(cur_vi_minus) or np.isnan(cur_atr)):
            self.prev_slope = cur_slope
            self.prev_vi_spread = cur_vi_spread
            return

        # Volume spike in last 3 bars
        recent_vol_spike = np.any(self._vol_spikes[max(0, i - 2):i + 1])
        cur_vol_dry = self._vol_dry[i]

        side, lots = context.position
        self.bars_since_last_scale += 1

        # ==================================================================
        # EXIT — Slope negative OR VI- > VI+ OR trailing stop
        # ==================================================================
        if lots > 0:
            # Update trailing stop
            if cur_atr > 0:
                candidate_stop = price - self.atr_trail_mult * cur_atr
                if candidate_stop > self.trail_stop:
                    self.trail_stop = candidate_stop

            slope_bearish = cur_slope < 0
            vortex_bearish = cur_vi_minus > cur_vi_plus
            stop_hit = price <= self.trail_stop

            if slope_bearish or vortex_bearish or stop_hit:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                self.bars_since_last_scale = 0
                self.prev_slope = cur_slope
                self.prev_vi_spread = cur_vi_spread
                return

        # ==================================================================
        # REDUCE — Volume dries up but slope still positive -> half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_vol_dry and cur_slope > 0:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                self.prev_slope = cur_slope
                self.prev_vi_spread = cur_vi_spread
                return

        # ==================================================================
        # ADD POSITION — slope increasing + VI spread widening + scale<3
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            # Profit prerequisite: floating profit >= 1 ATR
            profit_ok = price >= self.entry_price + cur_atr
            # Minimum 10 bar gap
            gap_ok = self.bars_since_last_scale >= 10
            # Slope increasing
            slope_accel = (not np.isnan(self.prev_slope)
                           and cur_slope > self.prev_slope
                           and cur_slope > 0)
            # VI spread widening
            spread_widening = (not np.isnan(self.prev_vi_spread)
                               and cur_vi_spread > self.prev_vi_spread
                               and cur_vi_spread > 0)

            if profit_ok and gap_ok and slope_accel and spread_widening:
                base_lots = self._calc_lots(context, price, cur_atr)
                factor = SCALE_FACTORS[self.position_scale]
                add_lots = max(1, int(base_lots * factor))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                self.prev_slope = cur_slope
                self.prev_vi_spread = cur_vi_spread
                return

        # ==================================================================
        # ENTRY — flat, slope > 0 + VI+ > VI- + volume spike in last 3 bars
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            slope_bullish = cur_slope > 0
            vortex_bullish = cur_vi_plus > cur_vi_minus
            volume_ok = recent_vol_spike

            if slope_bullish and vortex_bullish and volume_ok:
                entry_lots = self._calc_lots(context, price, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

        # Save for next bar
        self.prev_slope = cur_slope
        self.prev_vi_spread = cur_vi_spread

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, atr_val: float) -> int:
        """Size position based on ATR trailing stop distance.

        lots = (equity * 0.02) / (atr_trail_mult * ATR * contract_multiplier)
        """
        risk_per_trade = context.equity * 0.02
        distance = self.atr_trail_mult * atr_val

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
