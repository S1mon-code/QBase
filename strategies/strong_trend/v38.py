"""
Strong Trend Strategy v38 — HMA + Ergodic + A/D Line
=====================================================
Captures large trending moves (>100%, 6+ months) using:
  1. HMA          — fast trend direction (Hull Moving Average)
  2. Ergodic      — double-smoothed momentum oscillator
  3. A/D Line     — accumulation/distribution volume confirmation

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v38.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v38.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.hma import hma
from indicators.momentum.ergodic import ergodic
from indicators.volume.ad_line import ad_line
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV38(TimeSeriesStrategy):
    """HMA + Ergodic + A/D Line — long-only strong trend strategy."""

    name = "strong_trend_v38"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    hma_period: int = 20           # Hull Moving Average period
    ergo_short: int = 5            # Ergodic short EMA period
    ergo_long: int = 20            # Ergodic long EMA period
    ad_lookback: int = 10          # A/D Line rising/falling lookback
    atr_trail_mult: float = 4.5   # Trailing stop ATR multiplier

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        self._hma = None
        self._ergo_line = None
        self._ergo_signal = None
        self._ad = None
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

        self._hma = hma(closes, self.hma_period)
        self._ergo_line, self._ergo_signal = ergodic(closes, self.ergo_short, self.ergo_long)
        self._ad = ad_line(highs, lows, closes, volumes)
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
        cur_hma = self._hma[i]
        prev_hma = self._hma[i - 1]
        cur_ergo = self._ergo_line[i]
        cur_ergo_sig = self._ergo_signal[i]
        prev_ergo = self._ergo_line[i - 1]
        prev_ergo_sig = self._ergo_signal[i - 1]
        cur_atr = self._atr[i]
        price = context.close_raw

        # A/D Line rising/falling
        ad_rising = (
            i >= self.ad_lookback
            and not np.isnan(self._ad[i])
            and not np.isnan(self._ad[i - self.ad_lookback])
            and self._ad[i] > self._ad[i - self.ad_lookback]
        )
        ad_falling = (
            i >= self.ad_lookback
            and not np.isnan(self._ad[i])
            and not np.isnan(self._ad[i - self.ad_lookback])
            and self._ad[i] < self._ad[i - self.ad_lookback]
        )

        # HMA rising
        hma_rising = (
            not np.isnan(cur_hma)
            and not np.isnan(prev_hma)
            and cur_hma > prev_hma
        )

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_hma) or np.isnan(cur_ergo) or np.isnan(cur_ergo_sig)
                or np.isnan(cur_atr)):
            return

        side, lots = context.position

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — close < HMA OR Ergodic crosses below signal OR trailing stop
        # ==================================================================
        if lots > 0:
            ergo_cross_below = (
                not np.isnan(prev_ergo) and not np.isnan(prev_ergo_sig)
                and prev_ergo >= prev_ergo_sig
                and cur_ergo < cur_ergo_sig
            )
            if price < cur_hma or ergo_cross_below or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — A/D falling but HMA still rising → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if ad_falling and hma_rising:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — Ergodic > 0.5 + A/D still rising + scale<3
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            ergo_strong = cur_ergo > 0.5

            if profit_ok and bar_gap_ok and ergo_strong and ad_rising:
                base_lots = self._calc_lots(context, price, cur_atr)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, close > HMA + HMA rising + Ergodic > signal + A/D rising
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            above_hma = price > cur_hma
            ergo_bullish = cur_ergo > cur_ergo_sig

            if above_hma and hma_rising and ergo_bullish and ad_rising:
                entry_lots = self._calc_lots(context, price, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, cur_atr: float) -> int:
        """Size position based on ATR distance."""
        risk_per_trade = context.equity * 0.02
        distance = self.atr_trail_mult * cur_atr

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
