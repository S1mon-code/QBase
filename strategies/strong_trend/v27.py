"""
Strong Trend Strategy v27 — Ichimoku + MACD + Volume Momentum
==============================================================
Captures large trending moves (>100%, 6+ months) using:
  1. Ichimoku Cloud — trend direction & support/resistance
  2. MACD           — momentum confirmation via histogram
  3. Volume Momentum — conviction filter (buying pressure)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v27.py --symbols AG --freq 4h --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v27.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.ichimoku import ichimoku
from indicators.momentum.macd import macd
from indicators.volume.volume_momentum import volume_momentum
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV27(TimeSeriesStrategy):
    """Ichimoku + MACD + Volume Momentum — long-only strong trend strategy."""

    name = "strong_trend_v27"
    warmup = 60
    freq = "4h"

    # ----- Tunable parameters (5 total) -----
    tenkan: int = 9              # Ichimoku Tenkan-sen period
    kijun: int = 26              # Ichimoku Kijun-sen period
    macd_fast: int = 12          # MACD fast EMA
    macd_slow: int = 26          # MACD slow EMA
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
        lookback = max(self.warmup, 52 + self.kijun + 10, self.macd_slow + 20)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Compute indicators -----
        tenkan_sen, kijun_sen, senkou_a, senkou_b_line, _ = ichimoku(
            highs, lows, closes, self.tenkan, self.kijun,
        )
        _, _, histogram = macd(closes, self.macd_fast, self.macd_slow, signal=9)
        vol_mom = volume_momentum(volumes, period=14)
        atr_vals = atr(highs, lows, closes, period=14)

        # Current values — Ichimoku senkou spans are extended by displacement,
        # so the "current" cloud values at bar index n-1 are at senkou index n-1
        n = len(closes)
        cur_senkou_a = senkou_a[n - 1] if n - 1 < len(senkou_a) else np.nan
        cur_senkou_b = senkou_b_line[n - 1] if n - 1 < len(senkou_b_line) else np.nan
        cur_hist = histogram[-1]
        prev_hist = histogram[-2]
        cur_vol_mom = vol_mom[-1]
        cur_atr = atr_vals[-1]
        price = context.current_bar.close_raw
        cur_close = closes[-1]

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_senkou_a) or np.isnan(cur_senkou_b)
                or np.isnan(cur_hist) or np.isnan(prev_hist)
                or np.isnan(cur_vol_mom) or np.isnan(cur_atr)):
            return

        # Derived signals
        cloud_top = max(cur_senkou_a, cur_senkou_b)
        cloud_bottom = min(cur_senkou_a, cur_senkou_b)
        above_cloud = cur_close > cloud_top
        below_cloud = cur_close < cloud_bottom
        hist_positive = cur_hist > 0
        hist_increasing = cur_hist > prev_hist
        hist_strongly_negative = cur_hist < -abs(cur_atr * 0.01)  # scaled threshold

        side, lots = context.position

        # Update trailing stop (only ratchet upward)
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — close below cloud OR MACD histogram strongly negative
        #        OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if below_cloud or hist_strongly_negative or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — volume_momentum < 0.8 but still above cloud → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_vol_mom < 0.8 and above_cloud:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD POSITION — MACD histogram increasing + volume_momentum > 1.5
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS

            if profit_ok and bar_gap_ok and hist_increasing and cur_vol_mom > 1.5:
                base_lots = self._calc_lots(context, cur_atr)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, close above Ichimoku cloud + MACD hist > 0
        #         + volume_momentum > 1.0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            if above_cloud and hist_positive and cur_vol_mom > 1.0:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, atr_now: float) -> int:
        """Size position based on ATR-based trailing stop distance."""
        risk_per_trade = context.equity * 0.02
        risk_per_lot = self.atr_trail_mult * atr_now * self.contract_multiplier

        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
