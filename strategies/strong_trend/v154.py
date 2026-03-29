"""
QBase Strong Trend Strategy v154 — Gradient Trend Signal + Supertrend
======================================================================

策略简介：使用梯度趋势信号（基于价格变化率的平滑梯度）检测趋势方向，
         配合 Supertrend 确认方向后入场。

使用指标：
  - Gradient Trend Signal (period=20, smoothing=5): 平滑梯度趋势方向
  - Supertrend (period=10, mult=3.0): 趋势方向与动态支撑
  - ATR (period=14): trailing stop

进场条件：
  1. Gradient signal > 0（梯度向上）
  2. Supertrend direction == 1（上升趋势）
  3. 收盘价 > Supertrend line

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Gradient signal 转负 或 Supertrend 翻转

优点：梯度信号对噪声鲁棒，平滑后趋势判断稳定
缺点：平滑引入滞后
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.ml.gradient_trend import gradient_signal
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr


class StrongTrendV154(TimeSeriesStrategy):
    """梯度趋势信号 + Supertrend方向确认。"""
    name = "strong_trend_v154"
    warmup = 60
    freq = "daily"

    grad_period: int = 20
    grad_smooth: int = 5
    st_period: int = 10
    st_mult: float = 3.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._gradient = None
        self._st_line = None
        self._st_dir = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._gradient = gradient_signal(closes, period=self.grad_period, smoothing=self.grad_smooth)
        self._st_line, self._st_dir = supertrend(highs, lows, closes, period=self.st_period, mult=self.st_mult)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        grad_val = self._gradient[i]
        st_dir = self._st_dir[i]
        st_line = self._st_line[i]
        atr_val = self._atr[i]
        if np.isnan(grad_val) or np.isnan(st_dir) or np.isnan(st_line) or np.isnan(atr_val):
            return

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
        if side == 0 and grad_val > 0 and st_dir == 1 and price > st_line:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and (grad_val < 0 or st_dir == -1):
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
