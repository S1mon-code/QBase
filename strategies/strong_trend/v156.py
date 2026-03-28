"""
QBase Strong Trend Strategy v156 — Robust Z-Score + ADX
========================================================

策略简介：使用基于中位数/MAD的鲁棒Z分数检测价格偏离，配合ADX确认
         趋势强度后入场做多。

使用指标：
  - Robust Z-Score (period=60): 基于中位数和MAD的鲁棒标准化
  - ADX (period=14): 趋势强度
  - ATR (period=14): trailing stop

进场条件：
  1. Robust Z-Score > 0.5（价格在中位数上方偏离）
  2. ADX > 25（趋势足够强）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Robust Z-Score < -0.5（价格回归中位数以下）

优点：对离群值鲁棒，比标准Z-Score更可靠
缺点：MAD估计在小样本下可能不够精确
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.ml.robust_zscore import robust_zscore
from indicators.trend.adx import adx
from indicators.volatility.atr import atr


class StrongTrendV156(TimeSeriesStrategy):
    """鲁棒Z分数趋势偏离 + ADX趋势强度确认。"""
    name = "strong_trend_v156"
    warmup = 60
    freq = "daily"

    zscore_period: int = 60
    adx_period: int = 14
    adx_threshold: float = 25.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._zscore = None
        self._adx = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._zscore = robust_zscore(closes, period=self.zscore_period)
        self._adx = adx(highs, lows, closes, period=self.adx_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        z_val = self._zscore[i]
        adx_val = self._adx[i]
        atr_val = self._atr[i]
        if np.isnan(z_val) or np.isnan(adx_val) or np.isnan(atr_val):
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
        if side == 0 and z_val > 0.5 and adx_val > self.adx_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and z_val < -0.5:
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
