"""
Strong Trend Strategy v33 — T3 + Aroon + Volume Climax
=======================================================
Captures large trending moves (>100%, 6+ months) using:
  1. T3           — smooth trend direction & support level
  2. Aroon        — trend strength and momentum
  3. Volume Climax — buying pressure confirmation

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v33.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v33.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.t3 import t3
from indicators.trend.aroon import aroon
from indicators.volume.volume_spike import volume_climax
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV33(TimeSeriesStrategy):
    """T3 + Aroon + Volume Climax — long-only strong trend strategy."""

    name = "strong_trend_v33"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    t3_period: int = 5             # T3 smoothing period
    t3_vfactor: float = 0.7       # T3 volume factor
    aroon_period: int = 25         # Aroon lookback
    climax_period: int = 20        # Volume climax lookback
    atr_trail_mult: float = 4.5   # Trailing stop ATR multiplier

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
        lookback = max(self.warmup, self.aroon_period + 10, self.climax_period + 10)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Compute indicators -----
        t3_vals = t3(closes, self.t3_period, self.t3_vfactor)
        aroon_up, aroon_down, aroon_osc = aroon(highs, lows, self.aroon_period)
        climax_vals = volume_climax(highs, lows, closes, volumes, self.climax_period)
        atr_vals = atr(highs, lows, closes, period=14)

        # Current values
        cur_t3 = t3_vals[-1]
        prev_t3 = t3_vals[-2] if len(t3_vals) >= 2 else np.nan
        cur_aroon_up = aroon_up[-1]
        cur_aroon_osc = aroon_osc[-1]
        cur_atr = atr_vals[-1]
        price = context.current_bar.close_raw

        # T3 rising: current T3 > previous T3
        t3_rising = (
            not np.isnan(cur_t3)
            and not np.isnan(prev_t3)
            and cur_t3 > prev_t3
        )

        # Volume climax positive in last 3 bars
        recent_climax = False
        if len(climax_vals) >= 3:
            for offset in range(1, 4):  # last 3 bars
                val = climax_vals[-offset]
                if not np.isnan(val) and val > 0:
                    recent_climax = True
                    break

        # New volume climax (current bar)
        new_climax = (
            not np.isnan(climax_vals[-1])
            and climax_vals[-1] > 0
        )

        # Guard: skip if indicators aren't ready
        if np.isnan(cur_t3) or np.isnan(cur_aroon_osc) or np.isnan(cur_atr):
            return

        side, lots = context.position

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — close < T3 OR Aroon osc < -50 OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if price < cur_t3 or cur_aroon_osc < -50 or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — Aroon osc < 0 but T3 still rising → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_aroon_osc < 0 and t3_rising:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD POSITION — Aroon up = 100, new volume climax, scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            aroon_max = cur_aroon_up >= 100.0

            if profit_ok and bar_gap_ok and aroon_max and new_climax:
                base_lots = self._calc_lots(context, price, cur_t3)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, close > T3 + T3 rising + Aroon osc > 50 + recent climax
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            above_t3 = price > cur_t3
            aroon_ok = cur_aroon_osc > 50
            volume_ok = recent_climax

            if above_t3 and t3_rising and aroon_ok and volume_ok:
                entry_lots = self._calc_lots(context, price, cur_t3)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, t3_line: float) -> int:
        """Size position based on distance to T3 line."""
        risk_per_trade = context.equity * 0.02
        distance = abs(price - t3_line)

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
