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

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        lookback = max(self.warmup, self.hma_period + 20, self.ergo_long + self.ergo_short + 10)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Compute indicators -----
        hma_vals = hma(closes, self.hma_period)
        ergo_line, ergo_signal = ergodic(closes, self.ergo_short, self.ergo_long)
        ad_vals = ad_line(highs, lows, closes, volumes)
        atr_vals = atr(highs, lows, closes, period=14)

        # Current values
        cur_hma = hma_vals[-1]
        prev_hma = hma_vals[-2] if len(hma_vals) >= 2 else np.nan
        cur_ergo = ergo_line[-1]
        cur_ergo_sig = ergo_signal[-1]
        prev_ergo = ergo_line[-2] if len(ergo_line) >= 2 else np.nan
        prev_ergo_sig = ergo_signal[-2] if len(ergo_signal) >= 2 else np.nan
        cur_atr = atr_vals[-1]
        price = context.current_bar.close_raw

        # A/D Line rising/falling
        ad_rising = (
            len(ad_vals) > self.ad_lookback
            and not np.isnan(ad_vals[-1])
            and not np.isnan(ad_vals[-self.ad_lookback])
            and ad_vals[-1] > ad_vals[-self.ad_lookback]
        )
        ad_falling = (
            len(ad_vals) > self.ad_lookback
            and not np.isnan(ad_vals[-1])
            and not np.isnan(ad_vals[-self.ad_lookback])
            and ad_vals[-1] < ad_vals[-self.ad_lookback]
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
