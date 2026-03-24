"""
Strong Trend Strategy v14 — Donchian + RWI + Klinger
=====================================================
Captures large trending moves (>100%, 6+ months) using:
  1. Donchian Channel — breakout detection & trailing middle line
  2. RWI            — non-random trend confirmation
  3. Klinger (KVO)  — volume flow confirmation

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v14.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v14.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.donchian import donchian
from indicators.trend.rwi import rwi
from indicators.volume.klinger import klinger
from indicators.volatility.atr import atr


class StrongTrendV14(TimeSeriesStrategy):
    """Donchian + RWI + Klinger — long-only strong trend strategy."""

    name = "strong_trend_v14"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (5) -----
    don_period: int = 40            # Donchian channel lookback
    rwi_period: int = 14            # Random Walk Index lookback
    klinger_fast: int = 34          # Klinger fast EMA
    klinger_slow: int = 55          # Klinger slow EMA
    atr_trail_mult: float = 3.0     # ATR multiplier for trailing stop

    # ----- Position sizing -----
    contract_multiplier: float = 100.0  # Default for most commodities

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0     # 0 = flat, 1-3 = position tiers
        self.trail_stop = 0.0       # Trailing stop price

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        lookback = max(self.warmup, self.don_period + 10, self.klinger_slow + 20)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        # ----- Compute indicators -----
        don_upper, don_lower, don_mid = donchian(highs, lows, self.don_period)
        rwi_high, rwi_low = rwi(highs, lows, closes, self.rwi_period)
        kvo, kvo_signal = klinger(highs, lows, closes, volumes,
                                  fast=self.klinger_fast, slow=self.klinger_slow)
        atr_vals = atr(highs, lows, closes, period=14)

        # Current values
        price = context.current_bar.close_raw
        cur_don_upper = don_upper[-1]
        cur_don_mid = don_mid[-1]
        cur_rwi_high = rwi_high[-1]
        cur_kvo = kvo[-1]
        cur_kvo_sig = kvo_signal[-1]
        cur_atr = atr_vals[-1]
        prev_kvo = kvo[-2]
        prev_kvo_sig = kvo_signal[-2]

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_don_upper) or np.isnan(cur_rwi_high) or
                np.isnan(cur_kvo) or np.isnan(cur_kvo_sig) or np.isnan(cur_atr)):
            return

        side, lots = context.position

        # Update trailing stop if long
        if lots > 0 and cur_atr > 0:
            new_stop = price - self.atr_trail_mult * cur_atr
            if new_stop > self.trail_stop:
                self.trail_stop = new_stop

        # ==================================================================
        # EXIT — close < Donchian middle OR RWI high < 0.5 OR trailing stop
        # ==================================================================
        if lots > 0:
            if price < cur_don_mid or cur_rwi_high < 0.5 or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — KVO crosses below signal but RWI still > 1.0 → close half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            kvo_crossed_below = (prev_kvo >= prev_kvo_sig and cur_kvo < cur_kvo_sig)
            if kvo_crossed_below and cur_rwi_high > 1.0:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — RWI high > 1.5 + KVO still > signal + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            if cur_rwi_high > 1.5 and cur_kvo > cur_kvo_sig:
                add_lots = self._calc_lots(context, cur_atr)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                return

        # ==================================================================
        # ENTRY — close > Donchian upper + RWI high > 1.0 + KVO > signal
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            breakout = price > cur_don_upper
            trend_valid = cur_rwi_high > 1.0
            volume_bullish = cur_kvo > cur_kvo_sig

            if breakout and trend_valid and volume_bullish:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.trail_stop = price - self.atr_trail_mult * cur_atr

    # ------------------------------------------------------------------
    # Position sizing: equity * 0.02 / (atr_trail_mult * ATR * multiplier)
    # ------------------------------------------------------------------
    def _calc_lots(self, context, cur_atr: float) -> int:
        """Size position based on ATR trailing stop distance.

        lots = (equity * 0.02) / (atr_trail_mult * ATR * contract_multiplier)
        Ensures each lot risks roughly 2% of equity if price hits the
        trailing stop level.
        """
        risk_per_trade = context.equity * 0.02
        risk_per_lot = self.atr_trail_mult * cur_atr * self.contract_multiplier

        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))  # Clamp to [1, 50]
