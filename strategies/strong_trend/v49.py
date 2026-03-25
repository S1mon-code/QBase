"""
Strong Trend Strategy v49 — Hurst Exponent + Supertrend + Volume Spike
======================================================================
Regime-detection strategy: uses Hurst exponent to identify trending vs
mean-reverting regimes. Only trades when Hurst confirms a trending regime.

  1. Hurst Exponent — regime classification (>0.5 trending, <0.5 mean-reverting)
  2. Supertrend     — trend direction & trailing stop
  3. Volume Spike   — volume confirmation (demand surge)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v49.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v49.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.volatility.hurst import hurst_exponent
from indicators.trend.supertrend import supertrend
from indicators.volume.volume_spike import volume_spike
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV49(TimeSeriesStrategy):
    """Hurst Exponent + Supertrend + Volume Spike — regime-aware strong trend."""

    name = "strong_trend_v49"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    hurst_lag: int = 20            # Hurst R/S window
    st_period: int = 10            # Supertrend ATR lookback
    st_mult: float = 3.0           # Supertrend ATR multiplier
    vol_threshold: float = 2.0     # Volume spike threshold (x avg)
    atr_trail_mult: float = 4.5    # Trailing stop ATR multiplier

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        self._hurst = None
        self._st_line = None
        self._st_dir = None
        self._vol_spikes = None
        self._atr = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators once."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._hurst = hurst_exponent(closes, max_lag=self.hurst_lag)
        self._st_line, self._st_dir = supertrend(highs, lows, closes, self.st_period, self.st_mult)
        self._vol_spikes = volume_spike(volumes, period=20, threshold=self.vol_threshold)
        self._atr = atr(highs, lows, closes, period=14)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index
        self.bars_since_last_scale += 1

        # ----- Lookup pre-computed indicators -----
        cur_hurst = self._hurst[i]
        cur_dir = self._st_dir[i]
        cur_st_line = self._st_line[i]
        cur_atr = self._atr[i]
        price = context.close_raw

        # Volume spike in last 3 bars
        vol_spike_recent = bool(np.any(self._vol_spikes[max(0, i - 2):i + 1]))

        # Guard: skip if indicators aren't ready
        if np.isnan(cur_hurst) or np.isnan(cur_dir) or np.isnan(cur_st_line) or np.isnan(cur_atr):
            return

        side, lots = context.position

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — Supertrend flips bearish OR Hurst < 0.4 OR trailing stop
        # ==================================================================
        if lots > 0:
            st_bearish = cur_dir == -1
            regime_wrong = cur_hurst < 0.4  # strongly mean-reverting
            trail_hit = price < self.trail_stop

            if st_bearish or regime_wrong or trail_hit:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — Hurst drops below 0.5 but Supertrend still bullish → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            regime_shifting = cur_hurst < 0.5
            st_still_bullish = cur_dir == 1

            if regime_shifting and st_still_bullish:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — Hurst > 0.6 (strongly trending) + Supertrend bullish
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            hurst_strong = cur_hurst > 0.6
            st_bullish = cur_dir == 1

            if profit_ok and bar_gap_ok and hurst_strong and st_bullish:
                base_lots = self._calc_lots(context, price, cur_st_line)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — Hurst > 0.55 + Supertrend bullish + volume spike recent
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            regime_trending = cur_hurst > 0.55
            st_bullish = cur_dir == 1

            if regime_trending and st_bullish and vol_spike_recent:
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
