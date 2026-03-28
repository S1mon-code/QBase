"""
QBase Strong Trend Strategy v90 — Price Inertia + Volume Momentum
==================================================================

策略简介：Price Inertia 高值表示价格有惯性（趋势持续性），
         Volume Momentum 确认量能支撑后入场。

使用指标：
  - Price Inertia (period=20): 高值 = 价格惯性强 = 趋势
  - Volume Momentum (period=14): 量能动量确认
  - ATR (period=14): trailing stop

进场条件：
  1. Price Inertia > 0.6（高惯性 = 趋势 regime）
  2. Volume Momentum > 1.0（量能上升）
  3. 收盘价 > 前5日最高价（突破确认）

出场条件：
  1. ATR trailing stop（mult=4.0）
  2. Price Inertia < 0.3（惯性消失）

优点：Price Inertia 直接衡量价格动量的持续性
缺点：在价格震荡后突然启动时反应稍慢
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.price_inertia import price_inertia
from indicators.volume.volume_momentum import volume_momentum
from indicators.volatility.atr import atr


class StrongTrendV90(TimeSeriesStrategy):
    """High Price Inertia (trending) + Volume Momentum confirmation."""
    name = "strong_trend_v90"
    warmup = 60
    freq = "daily"

    inertia_threshold: float = 0.6
    vm_period: int = 14
    vm_threshold: float = 1.0
    breakout_lookback: int = 5
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._inertia = None
        self._vm = None
        self._atr = None
        self._highs = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._inertia = price_inertia(closes, period=20)
        self._vm = volume_momentum(volumes, self.vm_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._highs = highs

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        inertia_val = self._inertia[i]
        vm_val = self._vm[i]
        atr_val = self._atr[i]
        if np.isnan(inertia_val) or np.isnan(vm_val) or np.isnan(atr_val):
            return
        if i < self.breakout_lookback:
            return

        recent_high = np.max(self._highs[i - self.breakout_lookback:i])

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
        if side == 0 and inertia_val > self.inertia_threshold and vm_val > self.vm_threshold and price > recent_high:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and inertia_val < 0.3:
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
