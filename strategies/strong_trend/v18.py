"""
Strong Trend Strategy v18 — DEMA + KST + NR7
=============================================
Captures large trending moves using volatility-squeeze breakouts:
  1. DEMA  — trend direction & trailing reference
  2. KST   — momentum confirmation & acceleration
  3. NR7   — volatility squeeze detection (breakout filter)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v18.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v18.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.dema import dema
from indicators.momentum.kst import kst
from indicators.volatility.nr7 import nr7
from indicators.volatility.atr import atr


class StrongTrendV18(TimeSeriesStrategy):
    """DEMA + KST + NR7 — long-only squeeze-breakout trend strategy."""

    name = "strong_trend_v18"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (5) -----
    dema_period: int = 20
    kst_signal: int = 9
    nr7_lookback: int = 7       # NR7 uses fixed 7-bar window
    atr_period: int = 14
    atr_trail_mult: float = 3.0

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        self._dema = None
        self._kst_line = None
        self._kst_signal = None
        self._nr7 = None
        self._atr = None
        self._closes = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0       # 0 = flat, 1-3 = position tiers
        self.trailing_stop = 0.0      # ATR trailing stop level

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._closes = closes
        self._dema = dema(closes, self.dema_period)
        self._kst_line, self._kst_signal = kst(closes, signal_period=self.kst_signal)
        self._nr7 = nr7(highs, lows)
        self._atr = atr(highs, lows, closes, period=self.atr_period)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index
        if i < 4:
            return

        # Current values
        price = context.close_raw
        cur_close = self._closes[i]
        cur_dema = self._dema[i]
        prev_dema = self._dema[i - 1]
        cur_kst = self._kst_line[i]
        cur_kst_sig = self._kst_signal[i]
        cur_atr = self._atr[i]

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_dema) or np.isnan(cur_kst)
                or np.isnan(cur_kst_sig) or np.isnan(cur_atr)):
            return

        # Derived conditions
        dema_rising = cur_dema > prev_dema
        kst_bullish = cur_kst > cur_kst_sig
        recent_squeeze = np.any(self._nr7[i - 4:i + 1])  # NR7 in last 5 bars

        side, lots = context.position

        # ==================================================================
        # TRAILING STOP — update on every bar when long
        # ==================================================================
        if lots > 0 and self.trailing_stop > 0:
            new_stop = cur_close - self.atr_trail_mult * cur_atr
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop

        # ==================================================================
        # EXIT — close < DEMA OR KST < 0 OR trailing stop hit → close all
        # ==================================================================
        if lots > 0:
            stop_hit = cur_close <= self.trailing_stop
            below_dema = cur_close < cur_dema
            momentum_gone = cur_kst < 0

            if stop_hit or below_dema or momentum_gone:
                context.close_long()
                self.position_scale = 0
                self.trailing_stop = 0.0
                return

        # ==================================================================
        # REDUCE — KST crosses below signal but DEMA still rising → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if not kst_bullish and dema_rising:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — KST accelerating + still above DEMA + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            if i >= 2 and not np.isnan(self._kst_line[i - 2]):
                kst_accel = (self._kst_line[i] > self._kst_line[i - 1] > self._kst_line[i - 2])
                if kst_accel and cur_close > cur_dema:
                    add_lots = self._calc_lots(context, cur_atr)
                    if add_lots > 0:
                        context.buy(add_lots)
                        self.position_scale += 1
                        # Update trailing stop for new position
                        new_stop = cur_close - self.atr_trail_mult * cur_atr
                        self.trailing_stop = max(self.trailing_stop, new_stop)
                    return

        # ==================================================================
        # ENTRY — flat, breakout from squeeze with trend + momentum confirm
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            above_dema = cur_close > cur_dema

            if above_dema and dema_rising and kst_bullish and recent_squeeze:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.trailing_stop = cur_close - self.atr_trail_mult * cur_atr

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per ATR-based stop distance
    # ------------------------------------------------------------------
    def _calc_lots(self, context, cur_atr: float) -> int:
        """Size position: lots = (equity * 0.02) / (atr_trail_mult * ATR * multiplier)."""
        risk_per_trade = context.equity * 0.02
        stop_distance = self.atr_trail_mult * cur_atr

        if stop_distance <= 0:
            return 1

        risk_per_lot = stop_distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
