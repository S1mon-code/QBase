"""
Strong Trend Strategy v21 — Supertrend + ADX + OBV
====================================================
Captures large trending moves (>100%, 6+ months) using:
  1. Supertrend  — trend direction & trailing stop
  2. ADX         — trend strength filter
  3. OBV         — volume confirmation (buying pressure)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v21.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v21.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.supertrend import supertrend
from indicators.trend.adx import adx
from indicators.volume.obv import obv
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV21(TimeSeriesStrategy):
    """Supertrend + ADX + OBV — long-only strong trend strategy."""

    name = "strong_trend_v21"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    st_period: int = 10          # Supertrend ATR lookback
    st_mult: float = 3.0        # Supertrend ATR multiplier
    adx_period: int = 14         # ADX lookback
    adx_threshold: float = 25.0  # Min ADX to enter
    atr_trail_mult: float = 4.5  # Trailing stop ATR multiplier

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
        lookback = max(self.warmup, self.st_period + 10, self.adx_period * 3)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Compute indicators -----
        st_line, st_dir = supertrend(highs, lows, closes, self.st_period, self.st_mult)
        adx_vals = adx(highs, lows, closes, self.adx_period)
        obv_vals = obv(closes, volumes)
        atr_vals = atr(highs, lows, closes, period=14)

        # Current values
        cur_dir = st_dir[-1]
        cur_adx = adx_vals[-1]
        cur_st_line = st_line[-1]
        cur_atr = atr_vals[-1]
        price = context.current_bar.close_raw

        # OBV rising: current OBV > OBV 10 bars ago
        obv_rising = (
            len(obv_vals) >= 11
            and not np.isnan(obv_vals[-1])
            and not np.isnan(obv_vals[-11])
            and obv_vals[-1] > obv_vals[-11]
        )

        # Guard: skip if indicators aren't ready
        if np.isnan(cur_dir) or np.isnan(cur_adx) or np.isnan(cur_st_line) or np.isnan(cur_atr):
            return

        side, lots = context.position

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — Supertrend flips bearish OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if cur_dir == -1 or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — OBV falling but Supertrend still bullish → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if not obv_rising and cur_dir == 1:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD POSITION — already long, ADX > 35, OBV still rising
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            adx_strong = cur_adx > 35.0

            if profit_ok and bar_gap_ok and adx_strong and obv_rising:
                base_lots = self._calc_lots(context, price, cur_st_line)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, Supertrend bullish + ADX > threshold + OBV rising
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            trend_bullish = cur_dir == 1
            adx_ok = cur_adx > self.adx_threshold
            volume_ok = obv_rising

            if trend_bullish and adx_ok and volume_ok:
                entry_lots = self._calc_lots(context, price, cur_st_line)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, st_line: float) -> int:
        """Size position based on distance to Supertrend line."""
        risk_per_trade = context.equity * 0.02
        distance = abs(price - st_line)

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
