"""
Strong Trend Strategy v34 — McGinley Dynamic + PPO + OI Momentum
=================================================================
Captures large trending moves (>100%, 6+ months) using:
  1. McGinley Dynamic — adaptive trend-following line
  2. PPO              — percentage momentum confirmation
  3. OI Momentum      — open interest rate of change (positioning pressure)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v34.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v34.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.mcginley import mcginley_dynamic
from indicators.momentum.ppo import ppo
from indicators.volume.oi_momentum import oi_momentum
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV34(TimeSeriesStrategy):
    """McGinley Dynamic + PPO + OI Momentum — long-only strong trend strategy."""

    name = "strong_trend_v34"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    mcg_period: int = 14           # McGinley Dynamic smoothing period
    ppo_fast: int = 12             # PPO fast EMA
    ppo_slow: int = 26             # PPO slow EMA
    oi_period: int = 20            # OI momentum lookback
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
        lookback = max(self.warmup, self.ppo_slow + 20, self.oi_period + 10)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        oi_arr = context.get_volume_array(lookback)  # OI via volume array

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Compute indicators -----
        mcg_vals = mcginley_dynamic(closes, self.mcg_period)
        ppo_line, ppo_signal, ppo_hist = ppo(closes, self.ppo_fast, self.ppo_slow)
        oi_mom_vals = oi_momentum(oi_arr, self.oi_period)
        atr_vals = atr(highs, lows, closes, period=14)

        # Current values
        cur_mcg = mcg_vals[-1]
        prev_mcg = mcg_vals[-2] if len(mcg_vals) >= 2 else np.nan
        cur_ppo_hist = ppo_hist[-1]
        prev_ppo_hist = ppo_hist[-2] if len(ppo_hist) >= 2 else np.nan
        cur_oi_mom = oi_mom_vals[-1]
        cur_atr = atr_vals[-1]
        price = context.current_bar.close_raw

        # McGinley rising: current > previous
        mcg_rising = (
            not np.isnan(cur_mcg)
            and not np.isnan(prev_mcg)
            and cur_mcg > prev_mcg
        )

        # PPO histogram increasing
        ppo_hist_increasing = (
            not np.isnan(cur_ppo_hist)
            and not np.isnan(prev_ppo_hist)
            and cur_ppo_hist > prev_ppo_hist
        )

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_mcg) or np.isnan(cur_ppo_hist)
                or np.isnan(cur_oi_mom) or np.isnan(cur_atr)):
            return

        side, lots = context.position

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — close < McGinley OR PPO hist strongly negative OR trail stop
        # ==================================================================
        if lots > 0:
            ppo_strongly_negative = cur_ppo_hist < -0.5
            if price < cur_mcg or ppo_strongly_negative or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — OI momentum < 0 but McGinley still rising → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_oi_mom < 0 and mcg_rising:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD POSITION — PPO hist increasing + OI momentum > 5 + scale < 3
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            oi_strong = cur_oi_mom > 5.0

            if profit_ok and bar_gap_ok and ppo_hist_increasing and oi_strong:
                base_lots = self._calc_lots(context, price, cur_mcg)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, close > McGinley + rising + PPO hist > 0 + OI mom > 0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            above_mcg = price > cur_mcg
            ppo_ok = cur_ppo_hist > 0
            oi_ok = cur_oi_mom > 0

            if above_mcg and mcg_rising and ppo_ok and oi_ok:
                entry_lots = self._calc_lots(context, price, cur_mcg)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, mcg_line: float) -> int:
        """Size position based on distance to McGinley Dynamic line."""
        risk_per_trade = context.equity * 0.02
        distance = abs(price - mcg_line)

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
