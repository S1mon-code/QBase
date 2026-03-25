"""
Strong Trend Strategy v6 — Ichimoku + RSI + OBV
=================================================
Captures large trending moves (>100%, 6+ months) using:
  1. Ichimoku Cloud — trend direction, support/resistance, momentum (Tenkan/Kijun)
  2. RSI            — momentum confirmation & overbought/weakening filter
  3. OBV            — volume conviction (rising OBV = accumulation)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v6.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v6.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.ichimoku import ichimoku
from indicators.momentum.rsi import rsi
from indicators.volume.obv import obv
from indicators.volatility.atr import atr


class StrongTrendV6(TimeSeriesStrategy):
    """Ichimoku + RSI + OBV — long-only strong trend strategy."""

    name = "strong_trend_v6"
    warmup = 80
    freq = "daily"

    # ----- Tunable parameters (5) -----
    tenkan: int = 9              # Tenkan-sen period
    kijun: int = 26              # Kijun-sen period
    rsi_period: int = 14         # RSI lookback
    rsi_threshold: float = 50    # Min RSI for entry
    atr_trail_mult: float = 3.0  # Trailing stop ATR multiplier

    # ----- Position sizing -----
    contract_multiplier: float = 100.0  # Default for most commodities

    def __init__(self):
        super().__init__()
        self._tenkan_sen = None
        self._kijun_sen = None
        self._senkou_a = None
        self._senkou_b = None
        self._rsi = None
        self._obv = None
        self._atr = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0       # 0 = flat, 1-3 = position tiers
        self.highest_since_entry = 0.0  # Track highest price for trailing stop

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._tenkan_sen, self._kijun_sen, self._senkou_a, self._senkou_b, _ = ichimoku(
            highs, lows, closes, self.tenkan, self.kijun
        )
        self._rsi = rsi(closes, self.rsi_period)
        self._obv = obv(closes, volumes)
        self._atr = atr(highs, lows, closes, period=14)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index

        if i < 5:
            return

        # ----- Look up pre-computed indicators -----
        cur_tenkan = self._tenkan_sen[i]
        cur_kijun = self._kijun_sen[i]
        cur_senkou_a = self._senkou_a[i]
        cur_senkou_b = self._senkou_b[i]
        cur_rsi = self._rsi[i]
        cur_atr = self._atr[i]
        price = context.close_raw

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_tenkan) or np.isnan(cur_kijun) or np.isnan(cur_senkou_a)
                or np.isnan(cur_senkou_b) or np.isnan(cur_rsi) or np.isnan(cur_atr)):
            return

        # Derived signals
        above_cloud = price > cur_senkou_a and price > cur_senkou_b
        below_cloud = price < cur_senkou_a and price < cur_senkou_b
        tenkan_above_kijun = cur_tenkan > cur_kijun
        obv_rising = self._obv[i] > self._obv[i - 5]

        # Trailing stop level
        trailing_stop = self.highest_since_entry - self.atr_trail_mult * cur_atr

        side, lots = context.position

        # Update highest price since entry
        if lots > 0:
            if price > self.highest_since_entry:
                self.highest_since_entry = price

        # ==================================================================
        # EXIT — price drops below cloud OR trailing stop hit → close all
        # ==================================================================
        if lots > 0:
            if below_cloud or price < trailing_stop:
                context.close_long()
                self.position_scale = 0
                self.highest_since_entry = 0.0
                return

        # ==================================================================
        # REDUCE — RSI weakening but still above cloud → close half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_rsi < 45 and above_cloud:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — RSI > 60, new highs above cloud, scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            # Price making new highs: current close is highest in last 20 bars
            closes_arr = context.get_full_close_array()
            recent_start = max(0, i - 19)
            recent_high = np.max(closes_arr[recent_start:i + 1])
            new_highs = price >= recent_high

            if cur_rsi > 60 and new_highs and above_cloud:
                add_lots = self._calc_lots(context, cur_atr)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                return

        # ==================================================================
        # ENTRY — flat, above cloud + Tenkan > Kijun + RSI > threshold + OBV rising
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            if above_cloud and tenkan_above_kijun and cur_rsi > self.rsi_threshold and obv_rising:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.highest_since_entry = price

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, cur_atr: float) -> int:
        """Size position based on ATR trailing stop distance.

        lots = (equity * 0.02) / (atr_trail_mult * ATR * contract_multiplier)
        Ensures each lot risks roughly 2% of equity if price hits the
        trailing stop level.
        """
        risk_per_trade = context.equity * 0.02
        risk_distance = self.atr_trail_mult * cur_atr

        if risk_distance <= 0:
            return 1  # Fallback: minimum lot

        risk_per_lot = risk_distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))  # Clamp to [1, 50]
