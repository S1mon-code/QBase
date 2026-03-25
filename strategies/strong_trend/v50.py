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

    def __init__(self):
        super().__init__()
        self._yz = None
        self._yz_avg_40 = None
        self._don_upper = None
        self._don_mid = None
        self._obv = None
        self._atr = None

    def on_init(self, context):
        """Initialize tracking variables."""
        self.position_scale = 0
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.bars_since_last_scale = 999  # large initial value

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators once."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        opens = context.get_full_open_array()
        volumes = context.get_full_volume_array()

        self._yz = yang_zhang(opens, highs, lows, closes, period=self.yz_period)
        self._don_upper, _, self._don_mid = donchian(highs, lows, period=self.don_period)
        self._obv = obv(closes, volumes)
        self._atr = atr(highs, lows, closes, period=self.atr_period)

        # Pre-compute 40-bar rolling mean of YZ vol
        n = len(closes)
        self._yz_avg_40 = np.full(n, np.nan)
        for k in range(39, n):
            window = self._yz[k - 39:k + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                self._yz_avg_40[k] = np.mean(valid)

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------
    def on_bar(self, context):
        """Evaluate signals on every bar."""
        i = context.bar_index
        self.bars_since_last_scale += 1

        # ----- Lookup pre-computed indicators -----
        cur_yz = self._yz[i]
        cur_don_upper = self._don_upper[i]
        cur_don_mid = self._don_mid[i]
        cur_atr = self._atr[i]
        price = context.close_raw

        # YZ vol average over last 40 bars (pre-computed)
        yz_avg = self._yz_avg_40[i]

        # OBV rising: current OBV > OBV N bars ago
        obv_rising = (
            i >= self.obv_lookback
            and not np.isnan(self._obv[i])
            and not np.isnan(self._obv[i - self.obv_lookback])
            and self._obv[i] > self._obv[i - self.obv_lookback]
        )

        # OBV falling
        obv_falling = (
            i >= self.obv_lookback
            and not np.isnan(self._obv[i])
            and not np.isnan(self._obv[i - self.obv_lookback])
            and self._obv[i] < self._obv[i - self.obv_lookback]
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
