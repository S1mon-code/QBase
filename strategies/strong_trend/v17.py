"""
Strong Trend Strategy v17 — ALMA + Ultimate Oscillator + EMV
=============================================================
Captures large trending moves (>100%, 6+ months) using:
  1. ALMA               — trend direction & smoothed price filter
  2. Ultimate Oscillator — multi-timeframe momentum confirmation
  3. EMV                 — ease of movement (volume-weighted price effort)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v17.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v17.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.alma import alma
from indicators.momentum.ultimate_oscillator import ultimate_oscillator
from indicators.volume.emv import emv
from indicators.volatility.atr import atr


class StrongTrendV17(TimeSeriesStrategy):
    """ALMA + Ultimate Oscillator + EMV — long-only strong trend strategy."""

    name = "strong_trend_v17"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (5) -----
    alma_period: int = 9            # ALMA smoothing lookback
    alma_offset: float = 0.85       # ALMA Gaussian peak offset (0..1)
    uo_p1: int = 7                  # Ultimate Oscillator short period
    uo_p2: int = 14                 # Ultimate Oscillator medium period
    atr_trail_mult: float = 3.5     # ATR multiplier for trailing stop & sizing

    # ----- Fixed / derived -----
    _uo_p3: int = 28               # Ultimate Oscillator long period (2 * uo_p2)
    _atr_period: int = 14          # ATR lookback for trailing stop & sizing
    _alma_sigma: float = 6.0       # ALMA Gaussian width (default)
    contract_multiplier: float = 100.0

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0       # 0 = flat, 1-3 = position tiers
        self.trailing_stop = 0.0      # Trailing stop price
        self.highest_since_entry = 0.0

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        lookback = max(self.warmup, self._uo_p3 + 5, self._atr_period + 5)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        # ----- Compute indicators -----
        alma_arr = alma(closes, self.alma_period, self.alma_offset, self._alma_sigma)
        uo_arr = ultimate_oscillator(highs, lows, closes, self.uo_p1, self.uo_p2, self._uo_p3)
        emv_arr = emv(highs, lows, volumes, period=14)
        atr_arr = atr(highs, lows, closes, self._atr_period)

        # Current values
        cur_alma = alma_arr[-1]
        prev_alma = alma_arr[-2]
        cur_uo = uo_arr[-1]
        cur_emv = emv_arr[-1]
        cur_atr = atr_arr[-1]
        price = context.current_bar.close_raw

        # Guard: skip if indicators aren't ready
        if np.isnan(cur_alma) or np.isnan(prev_alma) or np.isnan(cur_uo) \
                or np.isnan(cur_emv) or np.isnan(cur_atr):
            return

        # Derived signals
        alma_rising = cur_alma > prev_alma
        above_alma = closes[-1] > cur_alma

        side, lots = context.position

        # Update trailing stop for open positions
        if lots > 0:
            self.highest_since_entry = max(self.highest_since_entry, price)
            self.trailing_stop = self.highest_since_entry - self.atr_trail_mult * cur_atr

        # ==================================================================
        # EXIT — close < ALMA OR UO < 30 OR trailing stop hit → close all
        # ==================================================================
        if lots > 0:
            if not above_alma or cur_uo < 30 or price <= self.trailing_stop:
                context.close_long()
                self.position_scale = 0
                self.trailing_stop = 0.0
                self.highest_since_entry = 0.0
                return

        # ==================================================================
        # REDUCE — EMV turns negative but ALMA still rising → close half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_emv < 0 and alma_rising:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — UO > 65 + EMV still positive + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            if cur_uo > 65 and cur_emv > 0:
                add_lots = self._calc_lots(context, cur_atr)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                return

        # ==================================================================
        # ENTRY — flat: close > ALMA + ALMA rising + UO > 50 + EMV > 0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            if above_alma and alma_rising and cur_uo > 50 and cur_emv > 0:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.highest_since_entry = price
                    self.trailing_stop = price - self.atr_trail_mult * cur_atr

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, cur_atr: float) -> int:
        """Size position based on ATR trailing-stop distance.

        lots = (equity * 0.02) / (atr_trail_mult * ATR * contract_multiplier)
        Ensures each lot risks roughly 2% of equity if price hits the
        trailing stop.
        """
        risk_per_trade = context.equity * 0.02
        distance = self.atr_trail_mult * cur_atr

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
