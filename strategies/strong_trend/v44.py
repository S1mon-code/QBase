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

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value
        self._prev_don_upper = np.nan     # track previous daily Donchian upper

    # ------------------------------------------------------------------
    # Aggregate 10min bars → daily bars (~24 bars per trading day)
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_daily(highs_10m, lows_10m, closes_10m, step=24):
        """Aggregate 10min OHLC arrays into approximate daily OHLC arrays."""
        n_groups = len(closes_10m) // step
        if n_groups < 1:
            return None, None, None
        trim = n_groups * step
        groups_c = closes_10m[:trim].reshape(n_groups, step)
        groups_h = highs_10m[:trim].reshape(n_groups, step)
        groups_l = lows_10m[:trim].reshape(n_groups, step)
        closes_d = groups_c[:, -1]
        highs_d = groups_h.max(axis=1)
        lows_d = groups_l.min(axis=1)
        return highs_d, lows_d, closes_d

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every 10min bar."""
        lookback = self.warmup
        closes_10m = context.get_close_array(lookback)
        highs_10m = context.get_high_array(lookback)
        lows_10m = context.get_low_array(lookback)

        if len(closes_10m) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Aggregate to daily -----
        highs_d, lows_d, closes_d = self._aggregate_daily(
            highs_10m, lows_10m, closes_10m, step=24
        )
        if highs_d is None or len(closes_d) < self.don_period + 5:
            return

        # ----- Compute daily Donchian channel -----
        don_upper, don_lower, don_mid = donchian(
            highs_d, lows_d, period=self.don_period
        )
        cur_don_upper = don_upper[-1]
        cur_don_mid = don_mid[-1]
        prev_don_upper = don_upper[-2] if len(don_upper) >= 2 else np.nan
        daily_close = closes_d[-1]

        if np.isnan(cur_don_upper) or np.isnan(cur_don_mid) or np.isnan(prev_don_upper):
            return

        # Detect new Donchian high: current daily close exceeds previous upper
        new_donchian_high = daily_close > prev_don_upper

        # ----- Compute 10min ROC -----
        roc_len = min(len(closes_10m), 200)
        roc_vals = rate_of_change(closes_10m[-roc_len:], period=self.roc_period)
        if len(roc_vals) == 0 or np.isnan(roc_vals[-1]):
            return
        cur_roc = roc_vals[-1]
        # ROC accelerating: current ROC > ROC 5 bars ago
        roc_accel = (
            len(roc_vals) >= 6
            and not np.isnan(roc_vals[-6])
            and cur_roc > roc_vals[-6]
        )

        # ----- Compute 10min ATR -----
        atr_vals = atr(
            highs_10m[-60:], lows_10m[-60:], closes_10m[-60:],
            period=self.atr_period,
        )
        if len(atr_vals) == 0 or np.isnan(atr_vals[-1]):
            return
        cur_atr = atr_vals[-1]

        price = context.current_bar.close_raw
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
