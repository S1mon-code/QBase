"""
Strong Trend Strategy v26 — TTM Squeeze + ADX + Force Index
=============================================================
Captures large trending moves using:
  1. TTM Squeeze  — volatility compression breakout + momentum
  2. ADX          — trend strength confirmation
  3. Force Index  — volume-weighted directional conviction

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v26.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v26.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.volatility.ttm_squeeze import ttm_squeeze
from indicators.trend.adx import adx
from indicators.volume.force_index import force_index
from indicators.volatility.atr import atr


# Pyramid scale factors: 100% -> 50% -> 25%
SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrongTrendV26(TimeSeriesStrategy):
    """TTM Squeeze + ADX + Force Index — long-only strong trend strategy."""

    name = "strong_trend_v26"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    squeeze_bb: int = 20         # TTM Squeeze Bollinger Band period
    squeeze_kc: int = 20         # TTM Squeeze Keltner Channel period
    adx_period: int = 14         # ADX lookback period
    fi_period: int = 13          # Force Index EMA period
    atr_trail_mult: float = 4.5  # ATR multiplier for trailing stop

    # ----- Position sizing -----
    contract_multiplier: float = 100.0

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0          # 0 = flat, 1-3 = position tiers
        self.entry_price = 0.0           # Average entry price
        self.trail_stop = 0.0            # Trailing stop level
        self.bars_since_last_scale = 0   # Min gap between adds
        self.prev_squeeze_on = False     # Previous bar squeeze state
        self.fi_median = 0.0             # Running median for "strongly positive"

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        lookback = max(self.warmup, self.squeeze_bb + 10,
                       self.adx_period * 3, self.fi_period + 10)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        # ----- Compute indicators -----
        squeeze_on, momentum = ttm_squeeze(
            highs, lows, closes,
            bb_period=self.squeeze_bb, kc_period=self.squeeze_kc,
        )
        adx_vals = adx(highs, lows, closes, period=self.adx_period)
        fi_vals = force_index(closes, volumes, period=self.fi_period)
        atr_vals = atr(highs, lows, closes, period=14)

        # Current values
        cur_squeeze = squeeze_on[-1]
        prev_squeeze = squeeze_on[-2]
        cur_momentum = momentum[-1]
        cur_adx = adx_vals[-1]
        cur_fi = fi_vals[-1]
        cur_atr = atr_vals[-1]
        price = context.current_bar.close_raw

        # Guard: skip if indicators aren't ready
        if (np.isnan(cur_momentum) or np.isnan(cur_adx)
                or np.isnan(cur_fi) or np.isnan(cur_atr)):
            self.prev_squeeze_on = cur_squeeze
            return

        # Compute Force Index "strongly positive" threshold (rolling)
        # Use recent positive FI values as reference
        fi_recent = fi_vals[-20:]
        fi_positive = fi_recent[~np.isnan(fi_recent) & (fi_recent > 0)]
        if len(fi_positive) > 0:
            self.fi_median = np.median(fi_positive)

        # Squeeze just released: was on, now off
        squeeze_released = self.prev_squeeze_on and not cur_squeeze

        side, lots = context.position
        self.bars_since_last_scale += 1

        # ==================================================================
        # EXIT — ADX < 15 OR momentum negative OR trailing stop
        # ==================================================================
        if lots > 0:
            # Update trailing stop (only tighten, never loosen)
            if cur_atr > 0:
                candidate_stop = price - self.atr_trail_mult * cur_atr
                if candidate_stop > self.trail_stop:
                    self.trail_stop = candidate_stop

            adx_weak = cur_adx < 15
            momentum_negative = cur_momentum < 0
            stop_hit = price <= self.trail_stop

            if adx_weak or momentum_negative or stop_hit:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                self.bars_since_last_scale = 0
                self.prev_squeeze_on = cur_squeeze
                return

        # ==================================================================
        # REDUCE — Force Index negative but ADX still strong -> half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_fi < 0 and cur_adx >= 20:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                self.prev_squeeze_on = cur_squeeze
                return

        # ==================================================================
        # ADD POSITION — ADX > 30 + Force Index strongly positive + scale<3
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            # Profit prerequisite: floating profit >= 1 ATR
            profit_ok = price >= self.entry_price + cur_atr
            # Minimum 10 bar gap
            gap_ok = self.bars_since_last_scale >= 10
            # ADX strong trend
            adx_strong = cur_adx > 30
            # Force Index strongly positive (above median of recent positives)
            fi_strong = cur_fi > self.fi_median and cur_fi > 0

            if profit_ok and gap_ok and adx_strong and fi_strong:
                base_lots = self._calc_lots(context, price, cur_atr)
                factor = SCALE_FACTORS[self.position_scale]
                add_lots = max(1, int(base_lots * factor))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                self.prev_squeeze_on = cur_squeeze
                return

        # ==================================================================
        # ENTRY — squeeze released + momentum > 0 + ADX > 20 + FI > 0
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            if (squeeze_released and cur_momentum > 0
                    and cur_adx > 20 and cur_fi > 0):
                entry_lots = self._calc_lots(context, price, cur_atr)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

        # Save for next bar
        self.prev_squeeze_on = cur_squeeze

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, atr_val: float) -> int:
        """Size position based on ATR trailing stop distance.

        lots = (equity * 0.02) / (atr_trail_mult * ATR * contract_multiplier)
        """
        risk_per_trade = context.equity * 0.02
        distance = self.atr_trail_mult * atr_val

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
