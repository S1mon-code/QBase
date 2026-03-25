"""
Strong Trend Strategy v7 — PSAR + TSI + CMF
=============================================
Captures large trending moves (>100%, 6+ months) using:
  1. Parabolic SAR — trend direction & trailing stop
  2. TSI           — double-smoothed momentum confirmation
  3. CMF           — volume-weighted buying/selling pressure

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v7.py --symbols AG --freq 4h --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v7.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.psar import psar
from indicators.momentum.tsi import tsi
from indicators.volume.cmf import cmf
from indicators.volatility.atr import atr


class StrongTrendV7(TimeSeriesStrategy):
    """PSAR + TSI + CMF — long-only strong trend strategy."""

    name = "strong_trend_v7"
    warmup = 60
    freq = "4h"

    # ----- Tunable parameters (5) -----
    psar_af_step: float = 0.02     # PSAR acceleration factor increment
    psar_af_max: float = 0.2       # PSAR max acceleration factor
    tsi_long: int = 25             # TSI long EMA period
    tsi_short: int = 13            # TSI short EMA period
    atr_trail_mult: float = 3.0    # ATR multiplier for trailing stop & sizing

    # ----- Position sizing -----
    contract_multiplier: float = 100.0  # Default for most commodities

    def __init__(self):
        super().__init__()
        self._psar_values = None
        self._psar_dir = None
        self._tsi_line = None
        self._tsi_signal = None
        self._cmf = None
        self._atr = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0        # 0 = flat, 1-3 = position tiers
        self.trailing_stop = 0.0       # Trailing stop price

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._psar_values, self._psar_dir = psar(
            highs, lows,
            af_start=self.psar_af_step,
            af_step=self.psar_af_step,
            af_max=self.psar_af_max,
        )
        self._tsi_line, self._tsi_signal = tsi(
            closes,
            long_period=self.tsi_long,
            short_period=self.tsi_short,
        )
        self._cmf = cmf(highs, lows, closes, volumes)
        self._atr = atr(highs, lows, closes)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index

        if i < 1:
            return

        # ----- Look up pre-computed indicators -----
        cur_psar_dir = self._psar_dir[i]
        prev_psar_dir = self._psar_dir[i - 1]
        cur_tsi = self._tsi_line[i]
        cur_tsi_sig = self._tsi_signal[i]
        cur_cmf = self._cmf[i]
        cur_atr = self._atr[i]
        price = context.close_raw

        # Guard: skip if indicators aren't ready
        if (
            np.isnan(cur_psar_dir)
            or np.isnan(cur_tsi)
            or np.isnan(cur_tsi_sig)
            or np.isnan(cur_cmf)
            or np.isnan(cur_atr)
        ):
            return

        side, lots = context.position

        # Update trailing stop for open positions
        if lots > 0 and cur_atr > 0:
            new_stop = price - self.atr_trail_mult * cur_atr
            self.trailing_stop = max(self.trailing_stop, new_stop)

        # ==================================================================
        # EXIT — PSAR flips bearish OR trailing stop hit → close everything
        # ==================================================================
        if lots > 0:
            psar_bearish = cur_psar_dir == -1
            stop_hit = price <= self.trailing_stop

            if psar_bearish or stop_hit:
                context.close_long()
                self.position_scale = 0
                self.trailing_stop = 0.0
                return

        # ==================================================================
        # REDUCE — CMF turns negative but PSAR still bullish → close half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_cmf < 0 and cur_psar_dir == 1:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — TSI strong + CMF positive + scale < 3 → buy more
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            if cur_tsi > 25 and cur_cmf > 0 and cur_psar_dir == 1:
                add_lots = self._calc_lots(context, cur_atr)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                return

        # ==================================================================
        # ENTRY — PSAR flips bullish + TSI > signal + TSI > 0 + CMF > 0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            psar_flip_bull = cur_psar_dir == 1 and prev_psar_dir == -1
            tsi_bullish = cur_tsi > cur_tsi_sig and cur_tsi > 0
            cmf_positive = cur_cmf > 0

            if psar_flip_bull and tsi_bullish and cmf_positive:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.trailing_stop = price - self.atr_trail_mult * cur_atr

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of ATR-based risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, cur_atr: float) -> int:
        """Size position based on ATR trailing stop distance.

        lots = (equity * 0.02) / (atr_trail_mult * ATR * contract_multiplier)
        Ensures each lot risks roughly 2% of equity if price moves
        atr_trail_mult * ATR against us.
        """
        risk_per_trade = context.equity * 0.02
        risk_distance = self.atr_trail_mult * cur_atr

        if risk_distance <= 0:
            return 1  # Fallback: minimum lot

        risk_per_lot = risk_distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))  # Clamp to [1, 50]
