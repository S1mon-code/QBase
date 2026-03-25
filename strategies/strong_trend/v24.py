"""
Strong Trend Strategy v24 — PSAR + CCI + OI Divergence
=======================================================
Captures large trending moves (>100%, 6+ months) using:
  1. Parabolic SAR — trend direction & trailing reference
  2. CCI           — momentum confirmation (>100 = strong uptrend)
  3. OI Divergence — open interest divergence for conviction

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v24.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v24.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.psar import psar
from indicators.momentum.cci import cci
from indicators.volume.oi_divergence import oi_divergence
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% -> 50% -> 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV24(TimeSeriesStrategy):
    """PSAR + CCI + OI Divergence — long-only strong trend strategy."""

    name = "strong_trend_v24"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (5) -----
    psar_af_step: float = 0.02   # PSAR acceleration factor step
    psar_af_max: float = 0.2     # PSAR max acceleration factor
    cci_period: int = 20          # CCI lookback period
    oi_period: int = 20           # OI Divergence lookback period
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
        lookback = max(self.warmup, self.cci_period + 5, self.oi_period + 10)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Compute indicators -----
        psar_vals, psar_dir = psar(
            highs, lows,
            af_step=self.psar_af_step, af_max=self.psar_af_max,
        )
        cci_vals = cci(highs, lows, closes, period=self.cci_period)
        atr_vals = atr(highs, lows, closes, period=14)

        # OI divergence — use actual OI if available, else volume as proxy
        try:
            oi_data = context.get_oi_array(lookback)
        except (AttributeError, Exception):
            oi_data = volumes  # Fallback: use volume as proxy
        oi_div = oi_divergence(closes, oi_data, period=self.oi_period)

        # Current values
        price = context.current_bar.close_raw
        cur_psar_dir = psar_dir[-1]
        cur_cci = cci_vals[-1]
        cur_oi_div = oi_div[-1]
        cur_atr = atr_vals[-1]

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_psar_dir) or np.isnan(cur_cci)
                or np.isnan(cur_oi_div) or np.isnan(cur_atr)):
            return

        side, lots = context.position

        # Update trailing stop if long
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — PSAR flips bearish OR CCI < -100 OR trailing stop hit
        # ==================================================================
        if lots > 0:
            psar_bearish = cur_psar_dir == -1
            cci_crash = cur_cci < -100
            trail_hit = price < self.trail_stop

            if psar_bearish or cci_crash or trail_hit:
                context.close_long()
                self._reset_state()
                return

        # ==================================================================
        # REDUCE — OI divergence negative but PSAR still bullish -> half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_oi_div < 0 and cur_psar_dir == 1:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD POSITION — CCI > 200 + OI divergence positive + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            cci_strong = cur_cci > 200
            oi_positive = cur_oi_div > 0

            if profit_ok and bar_gap_ok and cci_strong and oi_positive:
                base_lots = self._calc_lots(context, cur_atr)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, PSAR bullish + CCI > 100 + OI divergence > 0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            psar_bullish = cur_psar_dir == 1
            cci_ok = cur_cci > 100
            oi_ok = cur_oi_div > 0

            if psar_bullish and cci_ok and oi_ok:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _calc_lots(self, context, cur_atr: float) -> int:
        """Size position based on ATR trailing stop distance.

        lots = (equity * 0.02) / (atr_trail_mult * ATR * contract_multiplier)
        """
        risk_per_trade = context.equity * 0.02
        distance = self.atr_trail_mult * cur_atr

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))

    def _reset_state(self):
        """Clear tracking state after a full exit."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999
