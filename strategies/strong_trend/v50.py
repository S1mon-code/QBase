"""
Strong Trend Strategy v50 — Yang-Zhang Volatility + Donchian + OBV
===================================================================
Regime-detection strategy: strong trends are accompanied by volatility
EXPANSION. Uses Yang-Zhang vol to detect expanding regimes, Donchian
for breakout confirmation, and OBV for buying pressure.

  1. Yang-Zhang Volatility — regime detection (vol expansion = trending)
  2. Donchian Channel      — breakout confirmation
  3. OBV                   — volume/buying pressure confirmation

LONG ONLY. Supports add/reduce position scaling (0-3).

Usage:
    ./run.sh strategies/strong_trend/v50.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# QBase root (two levels up from strategies/strong_trend/v50.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401 — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

from indicators.volatility.yang_zhang import yang_zhang
from indicators.trend.donchian import donchian
from indicators.volume.obv import obv
from indicators.volatility.atr import atr


# ----- Pyramid scaling -----
SCALE_FACTORS = [1.0, 0.5, 0.25]  # 100% → 50% → 25%
MAX_SCALE = 3
MIN_BARS_BETWEEN_ADDS = 10


class StrongTrendV50(TimeSeriesStrategy):
    """Yang-Zhang Vol + Donchian + OBV — volatility-expansion regime trend."""

    name = "strong_trend_v50"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (<=5) -----
    yz_period: int = 20            # Yang-Zhang volatility period
    don_period: int = 40           # Donchian channel period
    obv_lookback: int = 10         # OBV rising comparison window
    atr_period: int = 14           # ATR period for trailing stop
    atr_trail_mult: float = 4.5    # Trailing stop ATR multiplier

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
        lookback = max(self.warmup, self.don_period + 10, self.yz_period + 42)
        closes = context.get_close_array(lookback)
        highs = context.get_high_array(lookback)
        lows = context.get_low_array(lookback)
        opens = context.get_open_array(lookback)
        volumes = context.get_volume_array(lookback)

        if len(closes) < lookback:
            return

        self.bars_since_last_scale += 1

        # ----- Compute indicators -----
        yz_vals = yang_zhang(opens, highs, lows, closes, period=self.yz_period)
        don_upper, don_lower, don_mid = donchian(highs, lows, period=self.don_period)
        obv_vals = obv(closes, volumes)
        atr_vals = atr(highs, lows, closes, period=self.atr_period)

        # Current values
        cur_yz = yz_vals[-1]
        cur_don_upper = don_upper[-1]
        cur_don_mid = don_mid[-1]
        cur_atr = atr_vals[-1]
        price = context.current_bar.close_raw

        # YZ vol average over last 40 bars for regime comparison
        yz_recent = yz_vals[-40:]
        yz_valid = yz_recent[~np.isnan(yz_recent)]
        yz_avg = np.mean(yz_valid) if len(yz_valid) > 0 else np.nan

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
        if (np.isnan(cur_yz) or np.isnan(yz_avg) or np.isnan(cur_don_upper)
                or np.isnan(cur_don_mid) or np.isnan(cur_atr)):
            return

        # Regime: vol expanding = YZ above its 40-bar average
        vol_expanding = cur_yz > yz_avg
        vol_collapsed = cur_yz < 0.5 * yz_avg

        # New Donchian high: current close is at or above the upper channel
        close_above_upper = price >= cur_don_upper

        side, lots = context.position

        # Update trailing stop
        if lots > 0 and cur_atr > 0:
            new_trail = price - self.atr_trail_mult * cur_atr
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail

        # ==================================================================
        # EXIT — Close < Donchian middle OR vol collapsed OR trailing stop
        # ==================================================================
        if lots > 0:
            below_mid = price < cur_don_mid
            trail_hit = price < self.trail_stop

            if below_mid or vol_collapsed or trail_hit:
                context.close_long()
                self.position_scale = 0
                self.entry_price = 0.0
                self.trail_stop = 0.0
                return

        # ==================================================================
        # REDUCE — OBV falling but YZ vol still high → half
        # ==================================================================
        if lots > 0 and self.position_scale > 0:
            if obv_falling and not vol_collapsed:
                half_lots = max(1, lots // 2)
                context.close_long(lots=half_lots)
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ==================================================================
        # ADD — Vol still expanding + new Donchian high + OBV rising
        # ==================================================================
        if lots > 0 and self.position_scale < MAX_SCALE:
            profit_ok = price >= self.entry_price + cur_atr  # profit >= 1 ATR
            bar_gap_ok = self.bars_since_last_scale >= MIN_BARS_BETWEEN_ADDS

            if profit_ok and bar_gap_ok and vol_expanding and close_above_upper and obv_rising:
                base_lots = self._calc_lots(context, price, cur_don_mid)
                add_lots = max(1, int(base_lots * SCALE_FACTORS[self.position_scale]))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
                return

        # ==================================================================
        # ENTRY — Vol expanding + close > Donchian upper + OBV rising
        # ==================================================================
        if lots == 0 and self.position_scale == 0:
            if vol_expanding and close_above_upper and obv_rising:
                entry_lots = self._calc_lots(context, price, cur_don_mid)
                if entry_lots > 0:
                    context.buy(entry_lots)
                    self.position_scale = 1
                    self.entry_price = price
                    self.trail_stop = price - self.atr_trail_mult * cur_atr
                    self.bars_since_last_scale = 0

    # ------------------------------------------------------------------
    # Position sizing: risk 2% of equity per unit of risk
    # ------------------------------------------------------------------
    def _calc_lots(self, context, price: float, reference_line: float) -> int:
        """Size position based on distance to Donchian middle."""
        risk_per_trade = context.equity * 0.02
        distance = abs(price - reference_line)

        if distance <= 0:
            return 1

        risk_per_lot = distance * self.contract_multiplier
        if risk_per_lot <= 0:
            return 1

        lots = int(risk_per_trade / risk_per_lot)
        return max(1, min(lots, 50))
