"""
QBase Strong Trend Strategy v172 — Random Projection Features + Chandelier Exit
=================================================================================

策略简介：随机投影将特征降维到低维空间，提取主要趋势信号，
         配合Chandelier Exit动态止损确认上升趋势。

使用指标：
  - Random Projection (n_components=5, period=120): 随机投影降维
  - Chandelier Exit (period=22, mult=3.0): 基于ATR的动态止损线
  - ATR (period=14): trailing stop

进场条件：
  1. 随机投影第一分量 > 0（主要趋势信号向上）
  2. 收盘价 > Chandelier Exit long线（上升趋势中）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. 收盘价 < Chandelier Exit long线

优点：随机投影计算快，对高维特征有效
缺点：随机性导致每次结果略有不同
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
from indicators.ml.random_projection import random_projection_features
from indicators.volatility.chandelier_exit import chandelier_exit
from indicators.volatility.atr import atr


class StrongTrendV172(TimeSeriesStrategy):
    """随机投影降维趋势信号 + Chandelier Exit动态止损。"""
    name = "strong_trend_v172"
    warmup = 60
    freq = "daily"

    rp_components: int = 5
    rp_period: int = 120
    ce_period: int = 22
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._rp_first = None
        self._ce_long = None
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

        rp = random_projection_features(features, n_components=self.rp_components, period=self.rp_period)
        self._rp_first = rp[:, 0] if rp.ndim > 1 else rp
        self._ce_long, _ = chandelier_exit(highs, lows, closes, period=self.ce_period, mult=3.0)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        rp_val = self._rp_first[i]
        ce_val = self._ce_long[i]
        atr_val = self._atr[i]
        if np.isnan(rp_val) or np.isnan(ce_val) or np.isnan(atr_val):
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
        if side == 0 and rp_val > 0 and price > ce_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and price < ce_val:
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
