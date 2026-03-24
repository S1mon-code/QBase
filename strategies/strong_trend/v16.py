"""
Strong Trend Strategy v16 — T3 + Ergodic + Twiggs Money Flow
=============================================================
Captures large trending moves (>100%, 6+ months) using:
  1. T3         — ultra-smooth trend filter & direction
  2. Ergodic    — double-smoothed momentum confirmation
  3. Twiggs MF  — volume-weighted conviction (gap-aware)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v16.py --symbols AG --freq 4h --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v16.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.t3 import t3
from indicators.momentum.ergodic import ergodic
from indicators.volume.twiggs import twiggs_money_flow
from indicators.volatility.atr import atr


class StrongTrendV16(TimeSeriesStrategy):
    """T3 + Ergodic + Twiggs Money Flow — long-only strong trend strategy."""

    name = "strong_trend_v16"
    warmup = 60
    freq = "4h"

    # ----- Tunable parameters (5) -----
    t3_period: int = 5              # T3 smoothing lookback
    t3_vfactor: float = 0.7        # T3 volume factor (0=EMA, 1=DEMA chain)
    ergo_short: int = 5            # Ergodic short EMA period
    ergo_long: int = 20            # Ergodic long EMA period
    atr_trail_mult: float = 3.0    # ATR multiplier for trailing stop

    # ----- Position sizing -----
    contract_multiplier: float = 100.0  # Default for most commodities

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0     # 0 = flat, 1-3 = position tiers
        self.trail_stop = 0.0       # Trailing stop level

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        lookback = max(self.warmup, self.ergo_long + self.ergo_short + 10)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        # ----- Compute indicators -----
        t3_line = t3(closes, period=self.t3_period, volume_factor=self.t3_vfactor)
        ergo_line, ergo_signal = ergodic(closes, short_period=self.ergo_short, long_period=self.ergo_long)
        tmf = twiggs_money_flow(highs, lows, closes, volumes)
        atr_vals = atr(highs, lows, closes, period=14)

        # Current values
        price = context.current_bar.close_raw
        cur_t3 = t3_line[-1]
        prev_t3 = t3_line[-2]
        cur_ergo = ergo_line[-1]
        cur_ergo_sig = ergo_signal[-1]
        prev_ergo = ergo_line[-2]
        prev_ergo_sig = ergo_signal[-2]
        cur_tmf = tmf[-1]
        cur_atr = atr_vals[-1]

        # Guard: skip if indicators aren't ready
        if any(np.isnan(v) for v in [cur_t3, prev_t3, cur_ergo, cur_ergo_sig,
                                      prev_ergo, prev_ergo_sig, cur_tmf, cur_atr]):
            return

        # Derived conditions
        t3_rising = cur_t3 > prev_t3
        close_above_t3 = price > cur_t3
        ergo_above_signal = cur_ergo > cur_ergo_sig
        ergo_cross_below = prev_ergo > prev_ergo_sig and cur_ergo <= cur_ergo_sig

        side, lots = context.position

        # Update trailing stop
        if lots > 0:
            new_stop = price - self.atr_trail_mult * cur_atr
            self.trail_stop = max(self.trail_stop, new_stop)

        # ==================================================================
        # EXIT — close < T3 OR Ergodic crosses below signal → close all
        # Also trailing stop hit
        # ==================================================================
        if lots > 0:
            if not close_above_t3 or ergo_cross_below or price <= self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — Twiggs MF turns negative but T3 still rising → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_tmf < 0 and t3_rising:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — Ergodic > 0.5 + Twiggs MF > 0.2 + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            if cur_ergo > 0.5 and cur_tmf > 0.2:
                add_lots = self._calc_lots(context, price, cur_atr)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                return

        # ==================================================================
        # ENTRY — flat: close > T3 + T3 rising + Ergodic > signal + TMF > 0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            if close_above_t3 and t3_rising and ergo_above_signal and cur_tmf > 0:
                entry_lots = self._calc_lots(context, price, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.trail_stop = price - self.atr_trail_mult * cur_atr

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, cur_atr: float) -> int:
        """Size position based on ATR trailing stop distance.

        lots = (equity * 0.02) / (atr_trail_mult * ATR * contract_multiplier)
        Ensures each lot risks roughly 2% of equity if price hits the
        trailing stop level.
        """
        risk_per_trade = context.equity * 0.02
        distance = self.atr_trail_mult * cur_atr

        if distance <= 0:
            return 1  # Fallback: minimum lot

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))  # Clamp to [1, 50]
