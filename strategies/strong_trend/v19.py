"""
Strong Trend Strategy v19 — McGinley Dynamic + Coppock Curve + A/D Line
========================================================================
Captures large trending moves (>100%, 6+ months) using:
  1. McGinley Dynamic — self-adjusting MA for trend direction
  2. Coppock Curve    — momentum confirmation (turns positive = buy)
  3. A/D Line         — volume/accumulation confirmation

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v19.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v19.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.mcginley import mcginley_dynamic
from indicators.momentum.coppock import coppock
from indicators.volume.ad_line import ad_line
from indicators.volatility.atr import atr


class StrongTrendV19(TimeSeriesStrategy):
    """McGinley Dynamic + Coppock Curve + A/D Line — long-only strong trend strategy."""

    name = "strong_trend_v19"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (5) -----
    mcg_period: int = 14            # McGinley Dynamic smoothing period
    cop_wma: int = 10               # Coppock WMA period
    cop_roc_long: int = 14          # Coppock long ROC lookback
    cop_roc_short: int = 11         # Coppock short ROC lookback
    atr_trail_mult: float = 3.0     # ATR multiplier for trailing stop

    # ----- Position sizing -----
    contract_multiplier: float = 100.0  # Default for most commodities

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0         # 0 = flat, 1-3 = position tiers
        self.trailing_stop = 0.0        # Trailing stop price
        self.prev_coppock = np.nan      # Previous bar's Coppock for acceleration check

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        lookback = max(self.warmup, self.cop_roc_long + self.cop_wma + 10)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        # ----- Compute indicators -----
        mcg = mcginley_dynamic(closes, self.mcg_period)
        cop = coppock(closes, self.cop_wma, self.cop_roc_long, self.cop_roc_short)
        ad = ad_line(highs, lows, closes, volumes)
        atr_vals = atr(highs, lows, closes, period=14)

        # Current values
        price = context.current_bar.close_raw
        cur_mcg = mcg[-1]
        prev_mcg = mcg[-2]
        cur_cop = cop[-1]
        prev_cop = cop[-2] if len(cop) >= 2 else np.nan
        cur_atr = atr_vals[-1]

        # Guard: skip if indicators aren't ready
        if np.isnan(cur_mcg) or np.isnan(cur_cop) or np.isnan(cur_atr):
            self.prev_coppock = cur_cop
            return

        # Derived signals
        mcg_rising = cur_mcg > prev_mcg
        cop_positive = cur_cop > 0
        cop_turns_positive = cur_cop > 0 and (not np.isnan(prev_cop) and prev_cop <= 0)
        cop_accelerating = (
            not np.isnan(self.prev_coppock)
            and cur_cop > self.prev_coppock
            and cur_cop > 0
        )
        ad_rising = ad[-1] > ad[-5] if len(ad) >= 5 else False
        ad_falling = not ad_rising

        side, lots = context.position

        # ==================================================================
        # Update trailing stop
        # ==================================================================
        if lots > 0 and not np.isnan(cur_atr):
            new_stop = price - self.atr_trail_mult * cur_atr
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop

        # ==================================================================
        # EXIT — close < McGinley OR Coppock turns negative OR trailing stop
        # ==================================================================
        if lots > 0:
            cop_negative = cur_cop < 0
            below_mcg = price < cur_mcg
            hit_trail = price <= self.trailing_stop

            if below_mcg or cop_negative or hit_trail:
                context.close_long()
                self.position_scale = 0
                self.trailing_stop = 0.0
                self.prev_coppock = cur_cop
                return

        # ==================================================================
        # REDUCE — A/D Line falling but McGinley still rising → close half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if ad_falling and mcg_rising:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                self.prev_coppock = cur_cop
                return

        # ==================================================================
        # ADD — Coppock accelerating + A/D still rising + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < 3:
            if cop_accelerating and ad_rising:
                add_lots = self._calc_lots(context, cur_atr)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                self.prev_coppock = cur_cop
                return

        # ==================================================================
        # ENTRY — flat, all conditions align
        # close > McGinley + McGinley rising + Coppock > 0 + A/D rising
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            above_mcg = price > cur_mcg
            if above_mcg and mcg_rising and cop_positive and ad_rising:
                entry_lots = self._calc_lots(context, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.trailing_stop = price - self.atr_trail_mult * cur_atr

        # Save Coppock for next bar's acceleration check
        self.prev_coppock = cur_cop

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
