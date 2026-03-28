"""
Strong Trend Strategy v116 — Vol-of-Vol Regime Change + EMA Ribbon
====================================================================
Detects transitions from stable to transitioning/crisis vol-of-vol regimes,
confirmed by EMA Ribbon alignment for trend direction.

  1. Vol-of-Vol Regime — 3-state regime classification (stable/transition/crisis)
  2. EMA Ribbon        — multi-period EMA alignment for trend strength

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v116.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.vol_of_vol_regime import vol_of_vol_regime
from indicators.trend.ema_ribbon import ema_ribbon_signal
from indicators.volatility.atr import atr


class StrongTrendV116(TimeSeriesStrategy):
    """
    策略简介：Vol-of-Vol状态转变 + EMA Ribbon趋势对齐的突破策略
    使用指标：Vol-of-Vol Regime（波动率的波动率状态）、EMA Ribbon（多周期均线排列）
    进场条件：从stable转变为transitioning + EMA Ribbon看多 + 价格上行
    出场条件：ATR trailing stop 或 EMA Ribbon转空
    优点：VoV状态转变是波动率爆发的前兆，EMA Ribbon确认方向
    缺点：状态转变可能短暂，crisis状态可能是崩盘
    """
    name = "strong_trend_v116"
    warmup = 60
    freq = "daily"

    vol_period: int = 20
    vov_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._vov = None
        self._is_stable = None
        self._is_trans = None
        self._ribbon = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
        self._prev_stable = False

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._vov, self._is_stable, self._is_trans, _ = vol_of_vol_regime(
            closes, vol_period=self.vol_period, vov_period=self.vov_period
        )
        self._ribbon = ema_ribbon_signal(closes, periods=[10, 20, 30, 50])
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        if i < 1:
            return

        is_stable = self._is_stable[i]
        is_trans = self._is_trans[i]
        prev_stable = self._is_stable[i - 1]
        ribbon_val = self._ribbon[i]
        atr_val = self._atr[i]

        if np.isnan(is_stable) or np.isnan(is_trans) or np.isnan(ribbon_val) or np.isnan(atr_val):
            return

        # Regime transition: was stable, now transitioning
        regime_shift = prev_stable == 1.0 and is_trans == 1.0
        ribbon_bullish = ribbon_val > 0

        # === Stop Loss Check ===
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # === Entry ===
        if side == 0 and regime_shift and ribbon_bullish:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and ribbon_val < 0:
            context.close_long()
            self._reset()

    def _calc_lots(self, context, price, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_trail_mult * atr_val * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = price * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        max_lots = int(context.equity * 0.30 / margin)
        return max(1, min(risk_lots, max_lots))

    def _reset(self):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
