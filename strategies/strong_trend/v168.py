"""
QBase Strong Trend Strategy v168 — Cross Validation Signal Strength + Donchian
================================================================================

策略简介：交叉验证信号强度评估模型预测的稳健性，当CV信号强且
         价格突破Donchian上轨时入场做多。

使用指标：
  - Cross Validation Signal (period=120, n_folds=3): CV稳健性评分
  - Donchian Channel (period=20): 通道突破
  - ATR (period=14): trailing stop

进场条件：
  1. CV signal strength > 0.5（模型在多折验证中表现稳定）
  2. 收盘价 > Donchian上轨（突破确认）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. 收盘价 < Donchian中轨（回落）

优点：交叉验证减少过拟合，信号更可靠
缺点：计算量大，信号产生频率低
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx
from indicators.ml.cross_validation_signal import cv_signal_strength
from indicators.trend.donchian import donchian
from indicators.volatility.atr import atr


class StrongTrendV168(TimeSeriesStrategy):
    """交叉验证信号强度 + Donchian通道突破。"""
    name = "strong_trend_v168"
    warmup = 60
    freq = "daily"

    cv_period: int = 120
    cv_folds: int = 3
    donchian_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._cv_signal = None
        self._dc_upper = None
        self._dc_mid = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        rsi_arr = rsi(closes, 14)
        adx_arr = adx(highs, lows, closes, 14)
        features = np.column_stack([rsi_arr, adx_arr])

        self._cv_signal = cv_signal_strength(closes, features, period=self.cv_period, n_folds=self.cv_folds)
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(highs, lows, period=self.donchian_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        cv_val = self._cv_signal[i]
        dc_up = self._dc_upper[i]
        dc_m = self._dc_mid[i]
        atr_val = self._atr[i]
        if np.isnan(cv_val) or np.isnan(dc_up) or np.isnan(dc_m) or np.isnan(atr_val):
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
        if side == 0 and cv_val > 0.5 and price > dc_up:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and price < dc_m:
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
