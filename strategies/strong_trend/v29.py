"""
Strong Trend Strategy v29 — ZLEMA + RSI + Klinger
===================================================
Captures large trending moves (>100%, 6+ months) using:
  1. ZLEMA    — zero-lag trend direction & trailing reference
  2. RSI      — momentum confirmation (positive, not overbought)
  3. Klinger  — volume-based conviction (KVO vs signal)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v29.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v29.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.zlema import zlema
from indicators.momentum.rsi import rsi
from indicators.volume.klinger import klinger
from indicators.volatility.atr import atr


# Pyramid scale factors: 100% → 50% → 25%
SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrongTrendV29(TimeSeriesStrategy):
    """ZLEMA + RSI + Klinger — long-only strong trend strategy."""

    name = "strong_trend_v29"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    zlema_period: int = 20
    rsi_period: int = 14
    klinger_fast: int = 34
    klinger_slow: int = 55
    atr_trail_mult: float = 4.5

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        self._zlema = None
        self._rsi = None
        self._kvo = None
        self._kvo_sig = None
        self._atr = None
        self._closes = None

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

        self._closes = closes
        self._zlema = zlema(closes, self.zlema_period)
        self._rsi = rsi(closes, self.rsi_period)
        self._kvo, self._kvo_sig = klinger(highs, lows, closes, volumes,
                                           fast=self.klinger_fast, slow=self.klinger_slow)
        self._atr = atr(highs, lows, closes, period=14)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index

        if i < 1:
            return

        # ----- Look up pre-computed indicators -----
        price = context.close_raw
        cur_zlema = self._zlema[i]
        prev_zlema = self._zlema[i - 1]
        cur_rsi = self._rsi[i]
        cur_kvo = self._kvo[i]
        cur_kvo_sig = self._kvo_sig[i]
        prev_kvo = self._kvo[i - 1]
        prev_kvo_sig = self._kvo_sig[i - 1]
        cur_atr = self._atr[i]

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_zlema) or np.isnan(prev_zlema) or np.isnan(cur_rsi)
                or np.isnan(cur_kvo) or np.isnan(cur_kvo_sig) or np.isnan(cur_atr)
                or np.isnan(prev_kvo) or np.isnan(prev_kvo_sig)):
            return

        # Derived signals
        zlema_rising = cur_zlema > prev_zlema
        close_above_zlema = self._closes[i] > cur_zlema
        kvo_bullish = cur_kvo > cur_kvo_sig
        kvo_strongly_bullish = cur_kvo > cur_kvo_sig + abs(cur_kvo_sig) * 0.3
        kvo_cross_below = prev_kvo > prev_kvo_sig and cur_kvo <= cur_kvo_sig

        side, lots = context.position
        self.bars_since_last_scale += 1

        # ==================================================================
        # EXIT — close < ZLEMA OR RSI < 30 OR trailing stop hit
        # ==================================================================
        if lots > 0:
            # Update trailing stop
            new_stop = price - self.atr_trail_mult * cur_atr
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop

            if (not close_above_zlema) or cur_rsi < 30 or price <= self.trailing_stop:
                context.close_long()
                self.position_scale = 0
                self.trailing_stop = 0.0
                self.entry_price = 0.0
                self.base_lots = 0
                self.bars_since_last_scale = 0
                return

        # ==================================================================
        # REDUCE — KVO crosses below signal but ZLEMA still rising → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if kvo_cross_below and zlema_rising:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — RSI > 60 + KVO strongly above signal + scale < 3
        # Requires: profit >= 1 ATR, min 10 bar gap
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr
            gap_ok = self.bars_since_last_scale >= 10

            if (profit_ok and gap_ok and cur_rsi > 60
                    and kvo_strongly_bullish):
                factor = SCALE_FACTORS[self.position_scale]
                add_lots = max(1, int(self.base_lots * factor))
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, close > ZLEMA + ZLEMA rising + RSI > 50 + KVO > signal
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            if close_above_zlema and zlema_rising and cur_rsi > 50 and kvo_bullish:
                entry_lots = self._calc_lots(context, price, cur_atr)
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
