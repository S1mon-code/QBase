"""
Strong Trend Strategy v1 — Supertrend + ROC + Volume Spike
============================================================
Captures large trending moves (>100%, 6+ months) using:
  1. Supertrend  — trend direction & trailing stop
  2. ROC         — momentum confirmation & acceleration
  3. Volume Spike — conviction filter

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v1.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v1.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.supertrend import supertrend
from indicators.momentum.roc import rate_of_change
from indicators.volume.volume_spike import volume_spike


class StrongTrendV1(TimeSeriesStrategy):
    """Supertrend + ROC + Volume Spike — long-only strong trend strategy."""

    name = "strong_trend_v1"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    st_period: int = 10          # Supertrend ATR lookback
    st_mult: float = 3.0        # Supertrend ATR multiplier
    roc_period: int = 20         # Rate of Change lookback
    roc_threshold: float = 5.0   # Min ROC% to enter
    vol_threshold: float = 2.0   # Volume spike detection threshold

    # ----- Position sizing -----
    contract_multiplier: float = 100.0  # Default for most commodities

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0     # 0 = flat, 1-3 = position tiers
        self.prev_roc = np.nan      # Previous bar's ROC for acceleration check

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        # Need enough bars for all indicators
        lookback = max(self.warmup, self.st_period + 10, self.roc_period + 5)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        # ----- Compute indicators -----
        st_line, st_dir = supertrend(highs, lows, closes, self.st_period, self.st_mult)
        roc = rate_of_change(closes, self.roc_period)
        vol_spikes = volume_spike(volumes, period=20, threshold=self.vol_threshold)

        # Current values
        cur_dir = st_dir[-1]
        prev_dir = st_dir[-2]
        cur_roc = roc[-1]
        cur_st_line = st_line[-1]
        price = context.current_bar.close_raw

        # Guard: skip if indicators aren't ready
        if np.isnan(cur_dir) or np.isnan(cur_roc) or np.isnan(cur_st_line):
            self.prev_roc = cur_roc
            return

        # Volume spike in last 3 bars
        recent_vol_spike = np.any(vol_spikes[-3:])

        side, lots = context.position

        # ==================================================================
        # EXIT — Supertrend flips bearish → close everything
        # ==================================================================
        if cur_dir == -1 and lots > 0:
            context.close_long()
            self.position_scale = 0
            self.prev_roc = cur_roc
            return

        # ==================================================================
        # REDUCE — ROC fading but trend still intact
        # Scale down when momentum weakens (ROC < half threshold)
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_roc < (self.roc_threshold / 2.0) and cur_dir == 1:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                self.prev_roc = cur_roc
                return

        # ==================================================================
        # ADD POSITION — already long, momentum accelerating, new spike
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            roc_accelerating = (
                not np.isnan(self.prev_roc)
                and self.prev_roc > 0
                and cur_roc > 1.5 * self.prev_roc
            )
            new_spike = vol_spikes[-1]  # Spike on this bar specifically

            if roc_accelerating and new_spike and cur_dir == 1:
                add_lots = self._calc_lots(context, price, cur_st_line)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                self.prev_roc = cur_roc
                return

        # ==================================================================
        # ENTRY — flat, Supertrend bullish, strong momentum, volume confirm
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            # Supertrend just flipped bullish OR already bullish while flat
            trend_bullish = cur_dir == 1
            momentum_ok = cur_roc > self.roc_threshold
            volume_ok = recent_vol_spike

            if trend_bullish and momentum_ok and volume_ok:
                entry_lots = self._calc_lots(context, price, cur_st_line)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1

        # Save ROC for next bar's acceleration check
        self.prev_roc = cur_roc

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
