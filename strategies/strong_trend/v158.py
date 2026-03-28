"""
QBase Strong Trend Strategy v158 — Wavelet Decompose (Trend Component) + HMA
==============================================================================

策略简介：小波分解提取价格趋势分量（低频部分），配合HMA趋势方向确认。

使用指标：
  - Wavelet Decompose (db4, level=4): 提取趋势分量
  - HMA (period=20): Hull移动平均线趋势方向
  - ATR (period=14): trailing stop

进场条件：
  1. Wavelet趋势分量上升（当前 > 前一bar）
  2. 收盘价 > HMA（价格在趋势线上方）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. 收盘价 < HMA

优点：小波分解能分离不同频率分量，趋势提取精确
缺点：边界效应在最近数据点可能不稳定
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.ml.wavelet_decompose import wavelet_features
from indicators.trend.hma import hma
from indicators.volatility.atr import atr


class StrongTrendV158(TimeSeriesStrategy):
    """小波分解趋势分量 + HMA方向确认。"""
    name = "strong_trend_v158"
    warmup = 60
    freq = "daily"

    wavelet_level: int = 4
    hma_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._wavelet_trend = None
        self._hma = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        wf = wavelet_features(closes, wavelet="db4", level=self.wavelet_level)
        # First column is the trend (approximation) component
        self._wavelet_trend = wf[:, 0] if wf.ndim > 1 else wf
        self._hma = hma(closes, period=self.hma_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        wt_val = self._wavelet_trend[i]
        wt_prev = self._wavelet_trend[i - 1] if i > 0 else np.nan
        hma_val = self._hma[i]
        atr_val = self._atr[i]
        if np.isnan(wt_val) or np.isnan(wt_prev) or np.isnan(hma_val) or np.isnan(atr_val):
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
        if side == 0 and wt_val > wt_prev and price > hma_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and price < hma_val:
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
