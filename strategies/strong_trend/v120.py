"""
Strong Trend Strategy v120 — Price Acceleration + Donchian Breakout
=====================================================================
Detects price acceleration (positive second derivative of price) combined
with Donchian channel breakout for high-conviction trend entries.

  1. Price Acceleration — 2nd derivative of price (momentum increasing)
  2. Donchian Channel   — breakout above upper channel = new highs

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v120.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.price_acceleration import price_acceleration
from indicators.trend.donchian import donchian
from indicators.volatility.atr import atr


class StrongTrendV120(TimeSeriesStrategy):
    """
    策略简介：价格加速度 + Donchian通道突破的趋势启动策略
    使用指标：Price Acceleration（价格二阶导数）、Donchian Channel（通道突破）
    进场条件：加速度为正 + 价格突破Donchian上轨
    出场条件：ATR trailing stop 或 价格跌破Donchian中轨
    优点：加速度捕捉动量增强的初期，Donchian是经典突破系统
    缺点：加速度波动大，需要平滑处理
    """
    name = "strong_trend_v120"
    warmup = 60
    freq = "daily"

    accel_period: int = 14
    don_period: int = 40
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._accel = None
        self._is_accel = None
        self._don_upper = None
        self._don_mid = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._accel, _, self._is_accel = price_acceleration(closes, period=self.accel_period)
        self._don_upper, _, self._don_mid = donchian(highs, lows, period=self.don_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        is_accel = self._is_accel[i]
        don_upper = self._don_upper[i]
        don_mid = self._don_mid[i]
        atr_val = self._atr[i]

        if np.isnan(is_accel) or np.isnan(don_upper) or np.isnan(atr_val):
            return

        accelerating = is_accel == 1.0
        above_donchian = price >= don_upper

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
        if side == 0 and accelerating and above_donchian:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and not np.isnan(don_mid) and price < don_mid:
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
