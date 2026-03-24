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

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0       # 0 = flat, 1-3 = position tiers
        self.trailing_stop = 0.0      # ATR trailing stop level

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        lookback = max(self.warmup, self.dema_period + 10, 50)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)

        if len(closes) < lookback:
            return

        # ----- Compute indicators -----
        dema_line = dema(closes, self.dema_period)
        kst_line, kst_signal = kst(closes, signal_period=self.kst_signal)
        nr7_flags = nr7(highs, lows)
        atr_vals = atr(highs, lows, closes, period=self.atr_period)

        # Current values
        price = context.current_bar.close_raw
        cur_close = closes[-1]
        cur_dema = dema_line[-1]
        prev_dema = dema_line[-2]
        cur_kst = kst_line[-1]
        cur_kst_sig = kst_signal[-1]
        cur_atr = atr_vals[-1]

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_dema) or np.isnan(cur_kst)
                or np.isnan(cur_kst_sig) or np.isnan(cur_atr)):
            return

        # Derived conditions
        dema_rising = cur_dema > prev_dema
        kst_bullish = cur_kst > cur_kst_sig
        recent_squeeze = np.any(nr7_flags[-5:])  # NR7 in last 5 bars

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
            if (len(kst_line) >= 3
                    and not np.isnan(kst_line[-3])):
                kst_accel = (kst_line[-1] > kst_line[-2] > kst_line[-3])
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
