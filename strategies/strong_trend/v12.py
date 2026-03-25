"""
Strong Trend Strategy v12 — Aroon + PPO + Volume Momentum
==========================================================
Captures large trending moves (>100%, 6+ months) using:
  1. Aroon      — trend direction & strength (oscillator)
  2. PPO        — momentum confirmation & acceleration
  3. Volume Mom — conviction filter (above-average volume)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v12.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v12.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.aroon import aroon
from indicators.momentum.ppo import ppo
from indicators.volume.volume_momentum import volume_momentum
from indicators.volatility.atr import atr


class StrongTrendV12(TimeSeriesStrategy):
    """Aroon + PPO + Volume Momentum — long-only strong trend strategy."""

    name = "strong_trend_v12"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (5) -----
    aroon_period: int = 25
    ppo_fast: int = 12
    ppo_slow: int = 26
    vol_mom_period: int = 14
    atr_trail_mult: float = 3.0

    # ----- Position sizing -----
    contract_multiplier: float = 100.0  # Default for most commodities

    def __init__(self):
        super().__init__()
        self._aroon_up = None
        self._aroon_down = None
        self._aroon_osc = None
        self._ppo_line = None
        self._ppo_signal = None
        self._ppo_hist = None
        self._vol_mom = None
        self._atr = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0       # 0 = flat, 1-3 = position tiers
        self.prev_ppo_hist = np.nan   # Previous bar's PPO histogram for acceleration
        self.trail_stop = 0.0         # Trailing stop level

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(highs, lows, self.aroon_period)
        self._ppo_line, self._ppo_signal, self._ppo_hist = ppo(closes, self.ppo_fast, self.ppo_slow)
        self._vol_mom = volume_momentum(volumes, self.vol_mom_period)
        self._atr = atr(highs, lows, closes, period=14)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index
        if i < 1:
            return

        # Current values
        cur_aroon_up = self._aroon_up[i]
        cur_aroon_osc = self._aroon_osc[i]
        cur_ppo_hist = self._ppo_hist[i]
        cur_vol_mom = self._vol_mom[i]
        cur_atr = self._atr[i]
        price = context.close_raw

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_aroon_osc) or np.isnan(cur_ppo_hist)
                or np.isnan(cur_vol_mom) or np.isnan(cur_atr)):
            self.prev_ppo_hist = cur_ppo_hist
            return

        side, lots = context.position

        # Update trailing stop for existing long positions
        if lots > 0 and cur_atr > 0:
            new_stop = price - self.atr_trail_mult * cur_atr
            self.trail_stop = max(self.trail_stop, new_stop)

        # ==================================================================
        # EXIT — Aroon oscillator < -50 (downtrend) OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if cur_aroon_osc < -50 or price <= self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.trail_stop = 0.0
                self.prev_ppo_hist = cur_ppo_hist
                return

        # ==================================================================
        # REDUCE — Volume momentum < 0.8 but Aroon still bullish → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_vol_mom < 0.8 and cur_aroon_osc > 0:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                self.prev_ppo_hist = cur_ppo_hist
                return

        # ==================================================================
        # ADD — Aroon up = 100 (new high) + PPO accelerating + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            prev_ppo_hist = self._ppo_hist[i - 1]
            ppo_accelerating = (
                not np.isnan(prev_ppo_hist)
                and cur_ppo_hist > 0
                and cur_ppo_hist > prev_ppo_hist
            )
            if cur_aroon_up >= 100 and ppo_accelerating:
                add_lots = self._calc_lots(context, cur_atr)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                self.prev_ppo_hist = cur_ppo_hist
                return

        # ==================================================================
        # ENTRY — flat, Aroon osc > 50 + PPO hist > 0 + vol mom > 1.0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            trend_strong = cur_aroon_osc > 50
            momentum_ok = cur_ppo_hist > 0
            volume_ok = cur_vol_mom > 1.0

            if trend_strong and momentum_ok and volume_ok:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    # Initialize trailing stop
                    self.trail_stop = price - self.atr_trail_mult * cur_atr

        # Save PPO histogram for next bar's acceleration check
        self.prev_ppo_hist = cur_ppo_hist

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
        distance = self.atr_trail_mult * cur_atr

        if distance <= 0:
            return 1  # Fallback: minimum lot

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))  # Clamp to [1, 50]
