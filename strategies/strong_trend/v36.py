"""
Strong Trend Strategy v36 — DEMA + Ultimate Oscillator + Twiggs Money Flow
===========================================================================
Captures large trending moves (>100%, 6+ months) using:
  1. DEMA              — trend direction & level (reduced-lag EMA)
  2. Ultimate Osc      — momentum across 3 timeframes [0-100]
  3. Twiggs Money Flow — volume-weighted buying/selling pressure

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v36.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v36.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.dema import dema
from indicators.momentum.ultimate_oscillator import ultimate_oscillator
from indicators.volume.twiggs import twiggs_money_flow
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV36(TimeSeriesStrategy):
    """DEMA + Ultimate Oscillator + Twiggs MF — long-only strong trend strategy."""

    name = "strong_trend_v36"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    dema_period: int = 20           # DEMA lookback
    uo_p1: int = 7                  # Ultimate Oscillator fast period
    uo_p2: int = 14                 # Ultimate Oscillator medium period
    uo_p3: int = 28                 # Ultimate Oscillator slow period
    atr_trail_mult: float = 4.5     # Trailing stop ATR multiplier

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
        lookback = max(self.warmup, self.uo_p3 + 10, self.dema_period * 3)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Compute indicators -----
        dema_vals = dema(closes, period=self.dema_period)
        uo_vals = ultimate_oscillator(highs, lows, closes,
                                      p1=self.uo_p1, p2=self.uo_p2, p3=self.uo_p3)
        tmf_vals = twiggs_money_flow(highs, lows, closes, volumes)
        atr_vals = atr(highs, lows, closes, period=14)

        # Current values
        cur_dema = dema_vals[-1]
        prev_dema = dema_vals[-2]
        cur_uo = uo_vals[-1]
        cur_tmf = tmf_vals[-1]
        cur_atr = atr_vals[-1]
        price = context.current_bar.close_raw

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_dema) or np.isnan(prev_dema) or np.isnan(cur_uo)
                or np.isnan(cur_tmf) or np.isnan(cur_atr)):
            return

        # Derived signals
        dema_rising = cur_dema > prev_dema

        side, lots = context.position

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — close < DEMA OR Ultimate Osc < 30 OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if price < cur_dema or cur_uo < 30.0 or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — Twiggs MF < 0 but DEMA still rising → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_tmf < 0 and dema_rising:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — already long, Ultimate Osc > 65, Twiggs MF > 0.2
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS

            if profit_ok and bar_gap_ok and cur_uo > 65.0 and cur_tmf > 0.2:
                base_lots = self._calc_lots(context, price, cur_dema)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, close > DEMA + DEMA rising + UO > 50 + TMF > 0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            trend_ok = price > cur_dema and dema_rising
            momentum_ok = cur_uo > 50.0
            volume_ok = cur_tmf > 0

            if trend_ok and momentum_ok and volume_ok:
                entry_lots = self._calc_lots(context, price, cur_dema)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, dema_line: float) -> int:
        """Size position based on distance to DEMA line."""
        risk_per_trade = context.equity * 0.02
        distance = abs(price - dema_line)

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
