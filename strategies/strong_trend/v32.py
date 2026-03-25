"""
Strong Trend Strategy v32 — EMA Ribbon + TSI + VROC
=====================================================
Captures large trending moves (>100%, 6+ months) using:
  1. EMA Ribbon  — trend alignment (bullish/bearish/mixed)
  2. TSI          — double-smoothed momentum confirmation
  3. VROC         — volume rate of change confirmation

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v32.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v32.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.ema_ribbon import ema_ribbon_signal
from indicators.momentum.tsi import tsi
from indicators.volume.vroc import vroc
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV32(TimeSeriesStrategy):
    """EMA Ribbon + TSI + VROC — long-only strong trend strategy."""

    name = "strong_trend_v32"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    ribbon_base: int = 8
    tsi_long: int = 25
    tsi_short: int = 13
    vroc_period: int = 14
    atr_trail_mult: float = 4.5

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        self._ribbon_sig = None
        self._tsi_line = None
        self._tsi_signal = None
        self._vroc = None
        self._atr = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        ribbon_periods = (
            self.ribbon_base,
            self.ribbon_base + 5,
            self.ribbon_base + 13,
            self.ribbon_base + 26,
            self.ribbon_base + 47,
        )
        self._ribbon_sig = ema_ribbon_signal(closes, ribbon_periods)
        self._tsi_line, self._tsi_signal = tsi(closes, self.tsi_long, self.tsi_short)
        self._vroc = vroc(volumes, self.vroc_period)
        self._atr = atr(highs, lows, closes, period=14)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index

        self.bars_since_last_scale += 1

        # ----- Lookup pre-computed indicators -----
        cur_ribbon = self._ribbon_sig[i]
        cur_tsi = self._tsi_line[i]
        cur_tsi_signal = self._tsi_signal[i]
        cur_vroc = self._vroc[i]
        cur_atr = self._atr[i]
        price = context.close_raw

        # Guard: skip if indicators aren't ready
        if (
            np.isnan(cur_ribbon)
            or np.isnan(cur_tsi)
            or np.isnan(cur_tsi_signal)
            or np.isnan(cur_vroc)
            or np.isnan(cur_atr)
        ):
            return

        side, lots = context.position

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — ribbon bearish OR TSI crosses below signal OR trailing stop
        # ==================================================================
        if lots > 0:
            ribbon_bearish = cur_ribbon == -1
            tsi_cross_below = cur_tsi < cur_tsi_signal
            trail_hit = price < self.trail_stop

            if ribbon_bearish or tsi_cross_below or trail_hit:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — VROC < -20 but ribbon still bullish → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_vroc < -20.0 and cur_ribbon == 1:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD POSITION — already long, TSI > 25, VROC > 20
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            tsi_strong = cur_tsi > 25.0
            vroc_strong = cur_vroc > 20.0

            if profit_ok and bar_gap_ok and tsi_strong and vroc_strong:
                base_lots = self._calc_lots(context, price, cur_atr)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, ribbon bullish + TSI > signal + TSI > 0 + VROC > 0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            ribbon_bullish = cur_ribbon == 1
            tsi_above_signal = cur_tsi > cur_tsi_signal
            tsi_positive = cur_tsi > 0
            vroc_positive = cur_vroc > 0

            if ribbon_bullish and tsi_above_signal and tsi_positive and vroc_positive:
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
        """Size position based on ATR-derived stop distance."""
        risk_per_trade = context.equity * 0.02
        distance = self.atr_trail_mult * cur_atr

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
