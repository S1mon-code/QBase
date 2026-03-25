"""
Strong Trend Strategy v40 — PSAR + KST + Klinger
==================================================
Captures large trending moves (>100%, 6+ months) using:
  1. PSAR    — trend direction & trailing reference
  2. KST     — multi-timeframe momentum confirmation
  3. Klinger — volume flow confirmation (buying pressure)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v40.py --symbols AG --freq 4h --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v40.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.psar import psar
from indicators.momentum.kst import kst
from indicators.volume.klinger import klinger
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV40(TimeSeriesStrategy):
    """PSAR + KST + Klinger — long-only strong trend strategy."""

    name = "strong_trend_v40"
    warmup = 60
    freq = "4h"

    # ----- Tunable parameters (<=5) -----
    psar_af_step: float = 0.02    # PSAR acceleration factor step
    psar_af_max: float = 0.2      # PSAR acceleration factor max
    kst_signal: int = 9           # KST signal line period
    klinger_fast: int = 34        # Klinger fast EMA period
    atr_trail_mult: float = 4.5   # Trailing stop ATR multiplier

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        self._psar_vals = None
        self._psar_dir = None
        self._kst_line = None
        self._kst_sig = None
        self._kvo_vals = None
        self._kvo_sig = None
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

        self._psar_vals, self._psar_dir = psar(
            highs, lows,
            af_start=self.psar_af_step, af_step=self.psar_af_step,
            af_max=self.psar_af_max,
        )
        self._kst_line, self._kst_sig = kst(closes, signal_period=self.kst_signal)
        self._kvo_vals, self._kvo_sig = klinger(
            highs, lows, closes, volumes, fast=self.klinger_fast,
        )
        self._atr = atr(highs, lows, closes, period=14)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index
        if i < 1:
            return

        self.bars_since_last_scale += 1

        # ----- Lookup pre-computed indicators -----
        cur_psar_dir = self._psar_dir[i]
        cur_psar_val = self._psar_vals[i]
        cur_kst = self._kst_line[i]
        cur_kst_sig = self._kst_sig[i]
        cur_kvo = self._kvo_vals[i]
        cur_kvo_sig = self._kvo_sig[i]
        cur_atr = self._atr[i]
        price = context.close_raw

        # Previous KST for acceleration check
        prev_kst = self._kst_line[i - 1]
        prev_kst_sig = self._kst_sig[i - 1]

        # Previous KVO signal for crossover detection
        prev_kvo = self._kvo_vals[i - 1]
        prev_kvo_sig = self._kvo_sig[i - 1]

        # Guard: skip if indicators aren't ready
        if (
            np.isnan(cur_psar_dir)
            or np.isnan(cur_kst)
            or np.isnan(cur_kst_sig)
            or np.isnan(cur_kvo)
            or np.isnan(cur_kvo_sig)
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
        # EXIT — PSAR flips bearish OR KST crosses below signal OR trailing
        # ==================================================================
        if lots > 0:
            psar_bearish = cur_psar_dir == -1
            kst_cross_down = (
                not np.isnan(prev_kst)
                and not np.isnan(prev_kst_sig)
                and prev_kst >= prev_kst_sig
                and cur_kst < cur_kst_sig
            )
            trail_hit = price < self.trail_stop

            if psar_bearish or kst_cross_down or trail_hit:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — KVO crosses below signal but PSAR still bullish → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            kvo_cross_down = (
                not np.isnan(prev_kvo)
                and not np.isnan(prev_kvo_sig)
                and prev_kvo >= prev_kvo_sig
                and cur_kvo < cur_kvo_sig
            )
            if kvo_cross_down and cur_psar_dir == 1:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD POSITION — KST accelerating + KVO strongly > signal
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS

            # KST accelerating: current spread wider than previous
            kst_accel = (
                not np.isnan(prev_kst)
                and not np.isnan(prev_kst_sig)
                and (cur_kst - cur_kst_sig) > (prev_kst - prev_kst_sig)
                and cur_kst > cur_kst_sig
            )
            # KVO strongly above signal
            kvo_strong = cur_kvo > cur_kvo_sig and (cur_kvo - cur_kvo_sig) > abs(cur_kvo_sig) * 0.1

            if profit_ok and bar_gap_ok and kst_accel and kvo_strong:
                base_lots = self._calc_lots(context, price, cur_psar_val)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, PSAR bullish + KST > signal + KVO > signal
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            psar_bullish = cur_psar_dir == 1
            kst_bullish = cur_kst > cur_kst_sig
            kvo_bullish = cur_kvo > cur_kvo_sig

            if psar_bullish and kst_bullish and kvo_bullish:
                entry_lots = self._calc_lots(context, price, cur_psar_val)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, psar_val: float) -> int:
        """Size position based on distance to PSAR line."""
        risk_per_trade = context.equity * 0.02
        distance = abs(price - psar_val)

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
