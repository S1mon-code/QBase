"""
Strong Trend Strategy v23 — Keltner Channel + ROC + Twiggs Money Flow
=====================================================================
Captures large trending moves (>100%, 6+ months) using:
  1. Keltner Channel — breakout above upper band confirms trend
  2. ROC             — momentum confirmation & acceleration
  3. Twiggs MF       — volume/money flow conviction filter

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v23.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v23.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.keltner import keltner
from indicators.momentum.roc import rate_of_change
from indicators.volume.twiggs import twiggs_money_flow
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% -> 50% -> 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV23(TimeSeriesStrategy):
    """Keltner Channel + ROC + Twiggs Money Flow — long-only strong trend strategy."""

    name = "strong_trend_v23"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (5) -----
    kc_ema: int = 20              # Keltner Channel EMA period
    kc_mult: float = 1.5         # Keltner Channel ATR multiplier
    roc_period: int = 20          # Rate of Change lookback
    roc_threshold: float = 5.0   # Min ROC% to enter
    atr_trail_mult: float = 4.5  # Trailing stop ATR multiplier

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def __init__(self):
        super().__init__()
        self._kc_upper = None
        self._kc_mid = None
        self._kc_lower = None
        self._roc = None
        self._tmf = None
        self._atr = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value
        self.prev_roc = np.nan            # previous bar ROC for acceleration

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._kc_upper, self._kc_mid, self._kc_lower = keltner(
            highs, lows, closes, ema_period=self.kc_ema, multiplier=self.kc_mult,
        )
        self._roc = rate_of_change(closes, self.roc_period)
        self._tmf = twiggs_money_flow(highs, lows, closes, volumes, period=21)
        self._atr = atr(highs, lows, closes, period=14)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index

        self.bars_since_last_scale += 1

        # ----- Look up pre-computed indicators -----
        price = context.close_raw
        cur_kc_upper = self._kc_upper[i]
        cur_kc_mid = self._kc_mid[i]
        cur_roc = self._roc[i]
        cur_tmf = self._tmf[i]
        cur_atr = self._atr[i]

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_kc_upper) or np.isnan(cur_kc_mid)
                or np.isnan(cur_roc) or np.isnan(cur_tmf) or np.isnan(cur_atr)):
            self.prev_roc = cur_roc
            return

        side, lots = context.position

        # Update trailing stop if long
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — close below Keltner middle OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if price < cur_kc_mid or price < self.trail_stop:
                context.close_long()
                self._reset_state()
                self.prev_roc = cur_roc
                return

        # ==================================================================
        # REDUCE — Twiggs MF negative but still above Keltner middle -> half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_tmf < 0 and price >= cur_kc_mid:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                self.prev_roc = cur_roc
                return

        # ==================================================================
        # ADD POSITION — ROC accelerating + Twiggs MF > 0.2 + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            roc_accelerating = (
                not np.isnan(self.prev_roc)
                and cur_roc > self.prev_roc
                and cur_roc > 0
            )
            tmf_strong = cur_tmf > 0.2

            if profit_ok and bar_gap_ok and roc_accelerating and tmf_strong:
                base_lots = self._calc_lots(context, cur_atr)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                self.prev_roc = cur_roc
                return

        # ==================================================================
        # ENTRY — flat, close > Keltner upper + ROC > threshold + TMF > 0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            breakout = price > cur_kc_upper
            momentum_ok = cur_roc > self.roc_threshold
            flow_ok = cur_tmf > 0

            if breakout and momentum_ok and flow_ok:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

        # Save ROC for next bar's acceleration check
        self.prev_roc = cur_roc

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
