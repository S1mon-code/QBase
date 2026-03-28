"""
QBase Strong Trend Strategy v157 — Trend Filter (L1) + PPO
============================================================

策略简介：使用L1趋势滤波（总变差正则化）提取分段线性趋势，
         配合PPO动量确认后入场。

使用指标：
  - Trend Filter (period=60, lambda=1.0): L1正则化趋势提取
  - PPO (fast=12, slow=26, signal=9): 百分比价格振荡器
  - ATR (period=14): trailing stop

进场条件：
  1. Trend Filter 斜率为正（当前 > 前一bar）
  2. PPO histogram > 0（动量确认）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. PPO histogram 转负

优点：L1滤波产生分段常数趋势，对突变敏感
缺点：lambda参数需要调优
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.ml.trend_filter import trend_filter
from indicators.momentum.ppo import ppo
from indicators.volatility.atr import atr


class StrongTrendV157(TimeSeriesStrategy):
    """L1趋势滤波 + PPO动量确认。"""
    name = "strong_trend_v157"
    warmup = 60
    freq = "daily"

    tf_period: int = 60
    tf_lambda: float = 1.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._trend = None
        self._ppo_hist = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._trend = trend_filter(closes, period=self.tf_period, lambda_param=self.tf_lambda)
        ppo_line, ppo_signal, ppo_hist = ppo(closes, fast=12, slow=26, signal=9)
        self._ppo_hist = ppo_hist
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        trend_val = self._trend[i]
        trend_prev = self._trend[i - 1] if i > 0 else np.nan
        ppo_val = self._ppo_hist[i]
        atr_val = self._atr[i]
        if np.isnan(trend_val) or np.isnan(trend_prev) or np.isnan(ppo_val) or np.isnan(atr_val):
            return

        trend_rising = trend_val > trend_prev

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
        if side == 0 and trend_rising and ppo_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and ppo_val < 0:
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
