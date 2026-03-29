"""
QBase Strong Trend Strategy v174 — Incremental PCA Signal + ALMA
==================================================================

策略简介：增量PCA在线更新主成分，实时生成趋势信号，
         配合ALMA平滑均线确认方向后入场。

使用指标：
  - Incremental PCA Signal (n_components=3): 在线PCA趋势信号
  - ALMA (period=9, offset=0.85, sigma=6): Arnaud Legoux移动平均
  - ATR (period=14): trailing stop

进场条件：
  1. Incremental PCA第一分量 > 0（主趋势向上）
  2. 收盘价 > ALMA（价格在平滑均线上方）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. 收盘价 < ALMA

优点：增量更新无需完整矩阵分解，计算高效
缺点：初始阶段主成分不稳定
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
from indicators.ml.incremental_pca import incremental_pca_signal
from indicators.trend.alma import alma
from indicators.volatility.atr import atr


class StrongTrendV174(TimeSeriesStrategy):
    """增量PCA在线趋势信号 + ALMA方向确认。"""
    name = "strong_trend_v174"
    warmup = 60
    freq = "daily"

    pca_components: int = 3
    alma_period: int = 9
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._pca_signal = None
        self._alma = None
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

        pca = incremental_pca_signal(features, n_components=self.pca_components)
        self._pca_signal = pca[:, 0] if pca.ndim > 1 else pca
        self._alma = alma(closes, period=self.alma_period, offset=0.85, sigma=6)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        pca_val = self._pca_signal[i]
        alma_val = self._alma[i]
        atr_val = self._atr[i]
        if np.isnan(pca_val) or np.isnan(alma_val) or np.isnan(atr_val):
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
        if side == 0 and pca_val > 0 and price > alma_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and price < alma_val:
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
