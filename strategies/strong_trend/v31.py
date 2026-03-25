"""
Strong Trend Strategy v31 — TEMA + Fisher Transform + OBV
==========================================================
Captures large trending moves (>100%, 6+ months) using:
  1. TEMA            — smoothed trend direction & trailing reference
  2. Fisher Transform — momentum confirmation (fisher vs trigger)
  3. OBV             — volume confirmation (buying pressure)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v31.py --symbols AG --freq 4h --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v31.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.tema import tema
from indicators.momentum.fisher_transform import fisher_transform
from indicators.volume.obv import obv
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV31(TimeSeriesStrategy):
    """TEMA + Fisher Transform + OBV — long-only strong trend strategy."""

    name = "strong_trend_v31"
    warmup = 60
    freq = "4h"

    # ----- Tunable parameters (<=5) -----
    tema_period: int = 20
    fisher_period: int = 10
    obv_lookback: int = 10
    atr_period: int = 14
    atr_trail_mult: float = 4.5

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
        lookback = max(self.warmup, self.tema_period * 3, self.fisher_period * 3)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Compute indicators -----
        tema_vals = tema(closes, self.tema_period)
        fisher_line, trigger_line = fisher_transform(highs, lows, self.fisher_period)
        obv_vals = obv(closes, volumes)
        atr_vals = atr(highs, lows, closes, period=self.atr_period)

        # Current values
        cur_tema = tema_vals[-1]
        prev_tema = tema_vals[-2]
        cur_fisher = fisher_line[-1]
        cur_trigger = trigger_line[-1]
        cur_atr = atr_vals[-1]
        price = context.current_bar.close_raw

        # TEMA rising: current > previous
        tema_rising = cur_tema > prev_tema

        # OBV rising: current OBV > OBV N bars ago
        obv_rising = (
            len(obv_vals) >= self.obv_lookback + 1
            and not np.isnan(obv_vals[-1])
            and not np.isnan(obv_vals[-(self.obv_lookback + 1)])
            and obv_vals[-1] > obv_vals[-(self.obv_lookback + 1)]
        )

        # OBV falling
        obv_falling = (
            len(obv_vals) >= self.obv_lookback + 1
            and not np.isnan(obv_vals[-1])
            and not np.isnan(obv_vals[-(self.obv_lookback + 1)])
            and obv_vals[-1] < obv_vals[-(self.obv_lookback + 1)]
        )

        # Guard: skip if indicators aren't ready
        if (
            np.isnan(cur_tema)
            or np.isnan(prev_tema)
            or np.isnan(cur_fisher)
            or np.isnan(cur_trigger)
            or np.isnan(cur_atr)
        ):
            return

        side, lots = context.position

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — close < TEMA OR Fisher crosses below trigger OR trailing stop
        # ==================================================================
        if lots > 0:
            price_below_tema = price < cur_tema
            fisher_cross_below = cur_fisher < cur_trigger
            trail_hit = price < self.trail_stop

            if price_below_tema or fisher_cross_below or trail_hit:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — OBV falling but TEMA still rising → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if obv_falling and tema_rising:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD POSITION — already long, Fisher > 1.0, OBV still rising
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            fisher_strong = cur_fisher > 1.0

            if profit_ok and bar_gap_ok and fisher_strong and obv_rising:
                base_lots = self._calc_lots(context, price, cur_tema)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, close > TEMA + TEMA rising + Fisher > trigger + OBV rising
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            price_above_tema = price > cur_tema
            fisher_bullish = cur_fisher > cur_trigger

            if price_above_tema and tema_rising and fisher_bullish and obv_rising:
                entry_lots = self._calc_lots(context, price, cur_tema)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, tema_line: float) -> int:
        """Size position based on distance to TEMA line."""
        risk_per_trade = context.equity * 0.02
        distance = abs(price - tema_line)

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
