"""
Strong Trend Strategy v44 — Daily Donchian + 10min ROC
======================================================
Multi-timeframe: daily Donchian channel identifies breakout direction,
10min Rate of Change confirms entry momentum.

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v44.py --symbols AG --freq 10min --start 2024
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v44.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.donchian import donchian
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV44(TimeSeriesStrategy):
    """Daily Donchian + 10min ROC — long-only strong trend strategy."""

    name = "strong_trend_v44"
    warmup = 1000
    freq = "10min"

    # ----- Tunable parameters (<=5) -----
    don_period: int = 40           # Donchian channel period (daily)
    roc_period: int = 20           # ROC period (10min)
    roc_threshold: float = 3.0     # Minimum ROC for entry
    atr_period: int = 14           # ATR lookback (10min)
    atr_trail_mult: float = 4.0   # Trailing stop ATR multiplier

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        # Small TF indicators (10min)
        self._roc = None
        self._atr = None
        # Large TF indicators (daily, pre-mapped to 10min indices)
        self._don_upper_daily = None
        self._don_lower_daily = None
        self._don_mid_daily = None
        self._closes_daily = None
        self._daily_map = None  # 10min index → daily index mapping

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value
        self._prev_don_upper = np.nan     # track previous daily Donchian upper

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators once. Aggregate 10min → daily."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        n = len(closes)

        # Small TF indicators (10min)
        self._roc = rate_of_change(closes, period=self.roc_period)
        self._atr = atr(highs, lows, closes, period=self.atr_period)

        # Aggregate to daily (step = 24: 24 × 10min ≈ 1 session)
        step = 24
        n_daily = n // step
        trim = n_daily * step
        closes_d = closes[:trim].reshape(n_daily, step)[:, -1]
        highs_d = highs[:trim].reshape(n_daily, step).max(axis=1)
        lows_d = lows[:trim].reshape(n_daily, step).min(axis=1)

        self._closes_daily = closes_d

        # Large TF indicators (daily Donchian)
        don_upper, don_lower, don_mid = donchian(
            highs_d, lows_d, period=self.don_period
        )
        self._don_upper_daily = don_upper
        self._don_lower_daily = don_lower
        self._don_mid_daily = don_mid

        # Index mapping: for 10min bar i, the latest COMPLETED daily bar
        self._daily_map = np.maximum(0, (np.arange(n) + 1) // step - 1)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every 10min bar."""
        i = context.bar_index
        j = self._daily_map[i]  # corresponding daily bar

        self.bars_since_last_scale += 1

        # ----- Lookup pre-computed values -----
        cur_roc = self._roc[i]
        cur_atr = self._atr[i]

        # NaN guard for small TF
        if np.isnan(cur_atr) or cur_atr <= 0:
            return

        # ----- Daily Donchian values -----
        cur_don_upper = self._don_upper_daily[j]
        cur_don_mid = self._don_mid_daily[j]
        daily_close = self._closes_daily[j]

        # Previous daily bar for Donchian high detection
        prev_don_upper = self._don_upper_daily[j - 1] if j >= 1 else np.nan

        if np.isnan(cur_don_upper) or np.isnan(cur_don_mid) or np.isnan(prev_don_upper):
            return

        # Detect new Donchian high: current daily close exceeds previous upper
        new_donchian_high = daily_close > prev_don_upper

        # ----- 10min ROC values -----
        if np.isnan(cur_roc):
            return

        # ROC accelerating: current ROC > ROC 5 bars ago
        roc_accel = (
            i >= 5
            and not np.isnan(self._roc[i - 5])
            and cur_roc > self._roc[i - 5]
        )

        price = context.close_raw
        side, lots = context.position

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — daily close < Donchian middle OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if daily_close < cur_don_mid or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — 10min ROC < 0 but daily still above Donchian middle → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_roc < 0.0 and daily_close >= cur_don_mid:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — new daily Donchian high + 10min ROC accelerating + scale<3
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS

            if profit_ok and bar_gap_ok and new_donchian_high and roc_accel:
                base_lots = self._calc_lots(context, price, cur_atr)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, daily broke above Donchian upper + 10min ROC > threshold
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            if new_donchian_high and cur_roc > self.roc_threshold:
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
    def _calc_lots(self, context, price: float, atr_val: float) -> int:
        """Size position based on ATR distance."""
        risk_per_trade = context.equity * 0.02
        distance = self.atr_trail_mult * atr_val

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
