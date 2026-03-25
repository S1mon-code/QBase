"""
Strong Trend Strategy v22 — HMA + Aroon + CMF
================================================
Captures large trending moves (>100%, 6+ months) using:
  1. HMA    — fast trend direction with minimal lag
  2. Aroon  — trend strength & new-high detection
  3. CMF    — buying/selling pressure confirmation

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v22.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v22.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.hma import hma
from indicators.trend.aroon import aroon
from indicators.volume.cmf import cmf
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV22(TimeSeriesStrategy):
    """HMA + Aroon + CMF — long-only strong trend strategy."""

    name = "strong_trend_v22"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    hma_period: int = 20          # HMA lookback
    aroon_period: int = 25        # Aroon lookback
    cmf_period: int = 20          # Chaikin Money Flow lookback
    atr_period: int = 14          # ATR for trailing stop
    atr_trail_mult: float = 4.5   # Trailing stop ATR multiplier

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        self._hma = None
        self._aroon_up = None
        self._aroon_down = None
        self._aroon_osc = None
        self._cmf = None
        self._atr = None
        self._closes = None

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

        self._closes = closes
        self._hma = hma(closes, self.hma_period)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(highs, lows, self.aroon_period)
        self._cmf = cmf(highs, lows, closes, volumes, self.cmf_period)
        self._atr = atr(highs, lows, closes, self.atr_period)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index

        if i < 1:
            return

        self.bars_since_last_scale += 1

        # ----- Look up pre-computed indicators -----
        price = context.close_raw
        cur_hma = self._hma[i]
        prev_hma = self._hma[i - 1]
        cur_aroon_osc = self._aroon_osc[i]
        cur_aroon_up = self._aroon_up[i]
        cur_cmf = self._cmf[i]
        cur_atr = self._atr[i]

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_hma) or np.isnan(prev_hma) or np.isnan(cur_aroon_osc)
                or np.isnan(cur_cmf) or np.isnan(cur_atr)):
            return

        # Derived signals
        hma_rising = cur_hma > prev_hma
        price_above_hma = self._closes[i] > cur_hma

        side, lots = context.position

        # Update trailing stop (only tighten, never loosen)
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — close < HMA OR Aroon osc < -50 OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if not price_above_hma or cur_aroon_osc < -50.0 or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — CMF < 0 but HMA still rising → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_cmf < 0.0 and hma_rising:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD POSITION — already long, Aroon up = 100, CMF still positive
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            aroon_perfect = cur_aroon_up >= 100.0
            cmf_positive = cur_cmf > 0.0

            if profit_ok and bar_gap_ok and aroon_perfect and cmf_positive:
                base_lots = self._calc_lots(context, price, cur_atr)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, close > HMA + HMA rising + Aroon osc > 50 + CMF > 0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            if price_above_hma and hma_rising and cur_aroon_osc > 50.0 and cur_cmf > 0.0:
                entry_lots = self._calc_lots(context, price, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk (ATR-based)
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, atr_val: float) -> int:
        """Size position based on ATR distance for stop.

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
