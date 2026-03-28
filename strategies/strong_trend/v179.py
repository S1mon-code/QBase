"""
QBase Strong Trend Strategy v179 — Seasonal Z-Score + ROC
===========================================================

策略简介：当价格高于季节性常态（季节性Z分数为正）且动量确认（ROC为正）时做多。

使用指标：
  - Seasonal Z-Score (period=252): 价格vs季节性均值的偏离度
  - ROC (period=12): 动量确认
  - ATR (period=14): trailing stop

进场条件：
  1. Seasonal Z-Score > 0.5（价格高于季节性常态）
  2. ROC > 0（动量为正）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. Seasonal Z-Score < -0.5（价格低于季节性常态）

优点：季节性偏离 + 动量双重验证
缺点：季节性Z分数需要足够历史数据
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.seasonality.seasonal_zscore import seasonal_zscore
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr


class StrongTrendV179(TimeSeriesStrategy):
    """季节性Z分数 + ROC动量策略。"""
    name = "strong_trend_v179"
    warmup = 60
    freq = "daily"

    seasonal_period: int = 252
    roc_period: int = 12
    sz_entry: float = 0.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._sz = None
        self._roc = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        datetimes = context.get_full_datetime_array()

        self._sz, _, _ = seasonal_zscore(closes, datetimes, period=self.seasonal_period)
        self._roc = rate_of_change(closes, period=self.roc_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        sz_val = self._sz[i]
        roc_val = self._roc[i]
        atr_val = self._atr[i]
        if np.isnan(sz_val) or np.isnan(roc_val) or np.isnan(atr_val):
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
        if side == 0 and sz_val > self.sz_entry and roc_val > 0.0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and sz_val < -0.5:
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
