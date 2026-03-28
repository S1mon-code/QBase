"""
QBase Strong Trend Strategy v99 — Mean Crossing Rate + CCI
============================================================

策略简介：Mean Crossing Rate 低值表示价格很少穿越均值（趋势持续），
         CCI 确认价格动量方向后入场。

使用指标：
  - Mean Crossing Rate (period=60): 低值 = 少穿越 = 趋势
  - CCI (period=20): > 100 表示强上升动量
  - ATR (period=14): trailing stop

进场条件：
  1. Mean Crossing Rate < 0.3（低穿越率 = 趋势 regime）
  2. CCI > 100（强动量）

出场条件：
  1. ATR trailing stop（mult=4.0）
  2. CCI < -100（动量反转）

优点：Mean Crossing Rate 简单直观，CCI 标准化处理好
缺点：CCI 在极端动量中可能过早触发出场
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.mean_crossing_rate import mean_crossing
from indicators.momentum.cci import cci
from indicators.volatility.atr import atr


class StrongTrendV99(TimeSeriesStrategy):
    """Low Mean Crossing Rate (trending) + CCI momentum."""
    name = "strong_trend_v99"
    warmup = 60
    freq = "daily"

    mcr_threshold: float = 0.3
    cci_period: int = 20
    cci_entry: float = 100.0
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._mcr = None
        self._cci = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._mcr = mean_crossing(closes, period=60)
        self._cci = cci(highs, lows, closes, period=self.cci_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        mcr_val = self._mcr[i]
        cci_val = self._cci[i]
        atr_val = self._atr[i]
        if np.isnan(mcr_val) or np.isnan(cci_val) or np.isnan(atr_val):
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
        if side == 0 and mcr_val < self.mcr_threshold and cci_val > self.cci_entry:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and cci_val < -100.0:
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
