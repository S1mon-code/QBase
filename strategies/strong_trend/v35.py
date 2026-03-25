"""
Strong Trend Strategy v35 — ALMA + Stochastic RSI + EMV
========================================================
Captures large trending moves (>100%, 6+ months) using:
  1. ALMA        — trend direction & level (Gaussian-weighted MA)
  2. Stoch RSI   — momentum confirmation (%K in [0,1])
  3. EMV         — volume ease-of-movement (positive = easy upward)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v35.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v35.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.alma import alma
from indicators.momentum.stoch_rsi import stoch_rsi
from indicators.volume.emv import emv
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV35(TimeSeriesStrategy):
    """ALMA + Stochastic RSI + EMV — long-only strong trend strategy."""

    name = "strong_trend_v35"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    alma_period: int = 9            # ALMA lookback
    alma_offset: float = 0.85       # ALMA gaussian peak position
    stochrsi_period: int = 14       # Stochastic RSI lookback
    emv_period: int = 14            # Ease of Movement smoothing period
    atr_trail_mult: float = 4.5     # Trailing stop ATR multiplier

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        self._alma = None
        self._k_line = None
        self._d_line = None
        self._emv = None
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

        self._alma = alma(closes, period=self.alma_period, offset=self.alma_offset)
        self._k_line, self._d_line = stoch_rsi(closes, rsi_period=self.stochrsi_period,
                                                stoch_period=self.stochrsi_period)
        self._emv = emv(highs, lows, volumes, period=self.emv_period)
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
        cur_alma = self._alma[i]
        prev_alma = self._alma[i - 1]
        cur_k = self._k_line[i]
        cur_emv = self._emv[i]
        cur_atr = self._atr[i]
        price = context.close_raw

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_alma) or np.isnan(prev_alma) or np.isnan(cur_k)
                or np.isnan(cur_emv) or np.isnan(cur_atr)):
            return

        # Derived signals
        alma_rising = cur_alma > prev_alma

        side, lots = context.position

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — close < ALMA OR StochRSI %K < 0.2 OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if price < cur_alma or cur_k < 0.2 or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — EMV < 0 but ALMA still rising → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_emv < 0 and alma_rising:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — already long, StochRSI %K > 0.8, EMV still positive
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS

            if profit_ok and bar_gap_ok and cur_k > 0.8 and cur_emv > 0:
                base_lots = self._calc_lots(context, price, cur_alma)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, close > ALMA + ALMA rising + StochRSI %K > 0.5 + EMV > 0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            trend_ok = price > cur_alma and alma_rising
            momentum_ok = cur_k > 0.5
            volume_ok = cur_emv > 0

            if trend_ok and momentum_ok and volume_ok:
                entry_lots = self._calc_lots(context, price, cur_alma)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, alma_line: float) -> int:
        """Size position based on distance to ALMA line."""
        risk_per_trade = context.equity * 0.02
        distance = abs(price - alma_line)

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
