"""
Strong Trend Strategy v9 — Supertrend + Stochastic RSI + Force Index
=====================================================================
Captures large trending moves using:
  1. Supertrend    — trend direction & trailing stop
  2. Stochastic RSI — momentum confirmation (not overbought filter)
  3. Force Index   — buying pressure / conviction filter

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v9.py --symbols AG --freq 1h --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v9.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.supertrend import supertrend
from indicators.momentum.stoch_rsi import stoch_rsi
from indicators.volume.force_index import force_index
from indicators.volatility.atr import atr


class StrongTrendV9(TimeSeriesStrategy):
    """Supertrend + Stochastic RSI + Force Index — long-only strong trend strategy."""

    name = "strong_trend_v9"
    warmup = 60
    freq = "1h"

    # ----- Tunable parameters (5) -----
    st_period: int = 10            # Supertrend ATR lookback
    st_mult: float = 3.0           # Supertrend ATR multiplier
    stochrsi_period: int = 14      # Stochastic RSI lookback (used for rsi & stoch)
    fi_period: int = 13            # Force Index EMA smoothing period
    atr_trail_mult: float = 3.0    # Trailing stop ATR multiplier

    # ----- Position sizing -----
    contract_multiplier: float = 100.0  # Default for most commodities

    def __init__(self):
        super().__init__()
        self._st_line = None
        self._st_dir = None
        self._k_line = None
        self._fi = None
        self._atr = None
        self._fi_avg_abs = 1.0  # Pre-computed mean of abs(force_index)

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0        # 0 = flat, 1-3 = position tiers
        self.trail_stop = 0.0          # Trailing stop price

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._st_line, self._st_dir = supertrend(highs, lows, closes, self.st_period, self.st_mult)
        self._k_line, _ = stoch_rsi(closes, rsi_period=self.stochrsi_period,
                                    stoch_period=self.stochrsi_period)
        self._fi = force_index(closes, volumes, period=self.fi_period)
        self._atr = atr(highs, lows, closes, period=self.st_period)

        # Pre-compute Force Index average (for "strongly positive" check)
        fi_valid = self._fi[~np.isnan(self._fi)]
        self._fi_avg_abs = np.mean(np.abs(fi_valid)) if len(fi_valid) > 0 else 1.0

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index

        # ----- Look up pre-computed indicators -----
        cur_dir = self._st_dir[i]
        cur_st_line = self._st_line[i]
        cur_k = self._k_line[i]
        cur_fi = self._fi[i]
        cur_atr = self._atr[i]
        price = context.close_raw

        # Guard: skip if indicators aren't ready
        if np.isnan(cur_dir) or np.isnan(cur_k) or np.isnan(cur_fi) or np.isnan(cur_atr):
            return

        fi_avg = self._fi_avg_abs

        side, lots = context.position

        # ----- Update trailing stop -----
        if lots > 0 and self.trail_stop > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — Supertrend flips bearish OR trailing stop hit -> close all
        # ==================================================================
        if lots > 0:
            trailing_hit = self.trail_stop > 0 and price < self.trail_stop
            if cur_dir == -1 or trailing_hit:
                context.close_long()
                self.position_scale = 0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — Force Index turns negative but Supertrend still bullish
        # -> close half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_fi < 0 and cur_dir == 1:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — StochRSI %K > 0.8 + Force Index strongly positive (> 2x avg)
        # + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            if cur_k > 0.8 and cur_fi > 2.0 * fi_avg and cur_dir == 1:
                add_lots = self._calc_lots(context, price, cur_st_line)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    # Reset trailing stop to accommodate new position
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                return

        # ==================================================================
        # ENTRY — flat, Supertrend bullish + StochRSI %K > 0.5 + FI > 0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            trend_bullish = cur_dir == 1
            momentum_ok = cur_k > 0.5
            buying_pressure = cur_fi > 0

            if trend_bullish and momentum_ok and buying_pressure:
                entry_lots = self._calc_lots(context, price, cur_st_line)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.trail_stop = price - self.atr_trail_mult * cur_atr

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, st_line: float) -> int:
        """Size position based on distance to Supertrend line.

        lots = (equity * 0.02) / (|price - supertrend| * contract_multiplier)
        Ensures each lot risks roughly 2% of equity if price touches the
        Supertrend stop level.
        """
        risk_per_trade = context.equity * 0.02
        distance = abs(price - st_line)

        if distance <= 0:
            return 1  # Fallback: minimum lot

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))  # Clamp to [1, 50]
