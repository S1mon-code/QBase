"""
Strong Trend Strategy v30 — Supertrend + Coppock Curve + CMF
=============================================================
Captures large trending moves (>100%, 6+ months) using:
  1. Supertrend     — trend direction & trailing stop
  2. Coppock Curve  — long-term momentum (buy signal from below zero)
  3. CMF            — buying/selling pressure confirmation

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v30.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v30.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.supertrend import supertrend
from indicators.momentum.coppock import coppock
from indicators.volume.cmf import cmf
from indicators.volatility.atr import atr


# Pyramid scale factors: 100% → 50% → 25%
SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrongTrendV30(TimeSeriesStrategy):
    """Supertrend + Coppock Curve + CMF — long-only strong trend strategy."""

    name = "strong_trend_v30"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    st_period: int = 10
    st_mult: float = 3.0
    cop_wma: int = 10
    cop_roc_long: int = 14
    atr_trail_mult: float = 4.5

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        self._st_line = None
        self._st_dir = None
        self._cop = None
        self._cmf = None
        self._atr = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0        # 0 = flat, 1-3 = position tiers
        self.entry_price = 0.0         # Average entry for profit check
        self.trailing_stop = 0.0       # Trailing stop level
        self.bars_since_last_scale = 0 # Min gap between adds
        self.base_lots = 0             # First entry lot size

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._st_line, self._st_dir = supertrend(highs, lows, closes, self.st_period, self.st_mult)
        self._cop = coppock(closes, wma_period=self.cop_wma, roc_long=self.cop_roc_long)
        self._cmf = cmf(highs, lows, closes, volumes)
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
        price = context.close_raw
        cur_dir = self._st_dir[i]
        cur_st_line = self._st_line[i]
        cur_cop = self._cop[i]
        prev_cop = self._cop[i - 1]
        cur_cmf = self._cmf[i]
        cur_atr = self._atr[i]

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_dir) or np.isnan(cur_st_line) or np.isnan(cur_cop)
                or np.isnan(prev_cop) or np.isnan(cur_cmf) or np.isnan(cur_atr)):
            return

        # Need 3 bars of coppock for acceleration check
        cop_m3 = self._cop[i - 2] if not np.isnan(self._cop[i - 2]) else np.nan

        # Derived signals
        st_bullish = cur_dir == 1
        st_bearish = cur_dir == -1
        cop_positive = cur_cop > 0
        cop_negative = cur_cop < 0
        cmf_positive = cur_cmf > 0
        cmf_negative = cur_cmf < 0
        cop_accelerating = (not np.isnan(cop_m3)
                            and cur_cop > prev_cop > cop_m3)

        side, lots = context.position
        self.bars_since_last_scale += 1

        # ==================================================================
        # EXIT — Supertrend bearish OR Coppock negative OR trailing stop
        # ==================================================================
        if lots > 0:
            # Update trailing stop
            new_stop = price - self.atr_trail_mult * cur_atr
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop

            if st_bearish or cop_negative or price <= self.trailing_stop:
                context.close_long()
                self.position_scale = 0
                self.trailing_stop = 0.0
                self.entry_price = 0.0
                self.base_lots = 0
                self.bars_since_last_scale = 0
                return

        # ==================================================================
        # REDUCE — CMF negative but Supertrend still bullish → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cmf_negative and st_bullish:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — Coppock accelerating + CMF positive + scale < 3
        # Requires: profit >= 1 ATR, min 10 bar gap
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr
            gap_ok = self.bars_since_last_scale >= 10

            if (profit_ok and gap_ok and cop_accelerating and cmf_positive):
                factor = SCALE_FACTORS[self.position_scale]
                add_lots = max(1, int(self.base_lots * factor))
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, Supertrend bullish + Coppock > 0 + CMF > 0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            if st_bullish and cop_positive and cmf_positive:
                entry_lots = self._calc_lots(context, price, cur_st_line, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.base_lots = entry_lots
                    self.trailing_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, st_line: float, atr_val: float) -> int:
        """Size position based on distance to Supertrend line.

        Uses the larger of Supertrend distance or ATR trailing stop distance.
        lots = (equity * 0.02) / (risk_distance * contract_multiplier)
        """
        risk_per_trade = context.equity * 0.02
        st_distance = abs(price - st_line)
        trail_distance = self.atr_trail_mult * atr_val
        distance = max(st_distance, trail_distance)

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
