"""
Strong Trend Strategy v43 — 4h Ichimoku Cloud + 5min Stochastic
================================================================
Multi-timeframe: 4h Ichimoku cloud sets trend direction,
5min Stochastic finds oversold dip-buy entries in uptrends.

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v43.py --symbols AG --freq 5min --start 2024
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v43.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.ichimoku import ichimoku
from indicators.momentum.stochastic import stochastic
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV43(TimeSeriesStrategy):
    """4h Ichimoku Cloud + 5min Stochastic — long-only strong trend strategy."""

    name = "strong_trend_v43"
    warmup = 2000
    freq = "5min"

    # ----- Tunable parameters (<=5) -----
    tenkan: int = 9              # Ichimoku Tenkan period (4h)
    kijun: int = 26              # Ichimoku Kijun period (4h)
    stoch_k: int = 14            # Stochastic %K period (5min)
    stoch_d: int = 3             # Stochastic %D smoothing (5min)
    atr_trail_mult: float = 4.0  # Trailing stop ATR multiplier (5min)

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value

    # ------------------------------------------------------------------
    # Aggregate 5min bars → 4h bars (48 bars per group)
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_4h(highs_5m, lows_5m, closes_5m, step=48):
        """Aggregate 5min OHLC arrays into 4h OHLC arrays."""
        n_groups = len(closes_5m) // step
        if n_groups < 1:
            return None, None, None
        trim = n_groups * step
        groups_c = closes_5m[:trim].reshape(n_groups, step)
        groups_h = highs_5m[:trim].reshape(n_groups, step)
        groups_l = lows_5m[:trim].reshape(n_groups, step)
        closes_4h = groups_c[:, -1]
        highs_4h = groups_h.max(axis=1)
        lows_4h = groups_l.min(axis=1)
        return highs_4h, lows_4h, closes_4h

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every 5min bar."""
        lookback = self.warmup
        closes_5m = context.get_close_array(lookback)
        highs_5m = context.get_high_array(lookback)
        lows_5m = context.get_low_array(lookback)

        if len(closes_5m) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Aggregate to 4h -----
        highs_4h, lows_4h, closes_4h = self._aggregate_4h(
            highs_5m, lows_5m, closes_5m, step=48
        )
        if highs_4h is None or len(closes_4h) < 60:
            return

        # ----- Compute 4h Ichimoku cloud -----
        tenkan_sen, kijun_sen, senkou_a, senkou_b_line, _ = ichimoku(
            highs_4h, lows_4h, closes_4h,
            tenkan=self.tenkan, kijun=self.kijun, senkou_b=52, displacement=26,
        )
        # Cloud top and bottom at the current 4h bar position
        # senkou_a/b are shifted forward by displacement, so current cloud
        # value for bar i is at index i in the extended array.
        n4h = len(closes_4h)
        cloud_idx = n4h - 1  # current 4h bar index in the extended arrays
        if cloud_idx >= len(senkou_a) or cloud_idx >= len(senkou_b_line):
            return
        sa = senkou_a[cloud_idx]
        sb = senkou_b_line[cloud_idx]
        if np.isnan(sa) or np.isnan(sb):
            return
        cloud_top = max(sa, sb)
        cloud_bottom = min(sa, sb)
        price_4h = closes_4h[-1]

        # ----- Compute 5min Stochastic -----
        stoch_len = min(len(closes_5m), 200)
        k_vals, d_vals = stochastic(
            highs_5m[-stoch_len:], lows_5m[-stoch_len:], closes_5m[-stoch_len:],
            k_period=self.stoch_k, d_period=self.stoch_d,
        )
        if len(k_vals) == 0 or np.isnan(k_vals[-1]):
            return
        cur_k = k_vals[-1]

        # ----- Compute 5min ATR -----
        atr_vals = atr(highs_5m[-60:], lows_5m[-60:], closes_5m[-60:], period=14)
        if len(atr_vals) == 0 or np.isnan(atr_vals[-1]):
            return
        cur_atr = atr_vals[-1]

        price = context.current_bar.close_raw
        side, lots = context.position

        # ----- Derived conditions -----
        above_cloud = price_4h > cloud_top
        below_cloud = price_4h < cloud_bottom

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — price drops below 4h cloud OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if below_cloud or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — Stochastic %K > 90 (overbought) → close half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_k > 90.0:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD POSITION — still above cloud + Stochastic %K < 15 + scale<3
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS

            if profit_ok and bar_gap_ok and above_cloud and cur_k < 15.0:
                base_lots = self._calc_lots(context, price, cur_atr)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, price above 4h cloud + Stochastic %K < 20
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            if above_cloud and cur_k < 20.0:
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
