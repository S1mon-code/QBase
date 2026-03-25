"""
Strong Trend Strategy v39 — Keltner Channel + CMO + Volume Spike
=================================================================
Captures large trending moves (>100%, 6+ months) using:
  1. Keltner Channel — breakout detection & trend reference
  2. CMO            — momentum strength filter [-100, 100]
  3. Volume Spike   — volume confirmation (demand surge)

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v39.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v39.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.trend.keltner import keltner
from indicators.momentum.cmo import cmo
from indicators.volume.volume_spike import volume_spike
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV39(TimeSeriesStrategy):
    """Keltner Channel + CMO + Volume Spike — long-only strong trend strategy."""

    name = "strong_trend_v39"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    kc_ema: int = 20              # Keltner EMA period
    kc_mult: float = 1.5          # Keltner ATR multiplier
    cmo_period: int = 14          # CMO lookback
    vol_threshold: float = 2.0    # Volume spike threshold (x avg)
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
        lookback = max(self.warmup, self.kc_ema + 20, self.cmo_period * 3)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Compute indicators -----
        kc_upper, kc_mid, kc_lower = keltner(
            highs, lows, closes, ema_period=self.kc_ema, multiplier=self.kc_mult,
        )
        cmo_vals = cmo(closes, period=self.cmo_period)
        vol_spikes = volume_spike(volumes, period=20, threshold=self.vol_threshold)
        atr_vals = atr(highs, lows, closes, period=14)

        # Current values
        cur_kc_upper = kc_upper[-1]
        cur_kc_mid = kc_mid[-1]
        cur_cmo = cmo_vals[-1]
        cur_atr = atr_vals[-1]
        price = context.current_bar.close_raw

        # Volume spike in last 3 bars
        has_recent_spike = bool(np.any(vol_spikes[-3:]))

        # Guard: skip if indicators aren't ready
        if (
            np.isnan(cur_kc_upper)
            or np.isnan(cur_kc_mid)
            or np.isnan(cur_cmo)
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
        # EXIT — close < Keltner middle OR CMO < -50 OR trailing stop hit
        # ==================================================================
        if lots > 0:
            if price < cur_kc_mid or cur_cmo < -50.0 or price < self.trail_stop:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — CMO < 0 but still above Keltner middle → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if cur_cmo < 0.0 and price >= cur_kc_mid:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD POSITION — already long, CMO > 50, new volume spike
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS
            cmo_strong = cur_cmo > 50.0

            if profit_ok and bar_gap_ok and cmo_strong and has_recent_spike:
                base_lots = self._calc_lots(context, price, cur_kc_mid)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — flat, close > Keltner upper + CMO > 20 + volume spike
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            breakout = closes[-1] > cur_kc_upper
            momentum_ok = cur_cmo > 20.0
            volume_ok = has_recent_spike

            if breakout and momentum_ok and volume_ok:
                entry_lots = self._calc_lots(context, price, cur_kc_mid)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, kc_mid: float) -> int:
        """Size position based on distance to Keltner middle."""
        risk_per_trade = context.equity * 0.02
        distance = abs(price - kc_mid)

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
