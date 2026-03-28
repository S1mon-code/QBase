"""
QBase Strong Trend Strategy v153 — Rolling PCA (1st Component Trend) + OI Momentum
====================================================================================

策略简介：滚动 PCA 提取第一主成分作为趋势信号，当第一主成分方向为正且
         持仓量动量上升时入场做多。

使用指标：
  - Rolling PCA (period=60, n_components=3): 第一主成分趋势信号
  - OI Momentum (period=20): 持仓量动量
  - ATR (period=14): trailing stop

进场条件：
  1. PCA 第一主成分 > 0（趋势向上）
  2. OI Momentum > 0（持仓量增加，资金流入）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. PCA 第一主成分转负

优点：PCA 降维捕捉主要趋势方向，不受噪声干扰
缺点：PCA 对输入特征敏感，需要合理构建特征矩阵
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
from indicators.ml.pca_features import rolling_pca
from indicators.volume.oi_momentum import oi_momentum
from indicators.volatility.atr import atr


class StrongTrendV153(TimeSeriesStrategy):
    """Rolling PCA第一主成分趋势 + OI动量确认。"""
    name = "strong_trend_v153"
    warmup = 60
    freq = "daily"

    pca_period: int = 60
    oi_mom_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._pca_first = None
        self._oi_mom = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        oi = context.get_full_oi_array()

        rsi_arr = rsi(closes, 14)
        adx_arr = adx(highs, lows, closes, 14)
        features = np.column_stack([rsi_arr, adx_arr])

        pca_result = rolling_pca(features, period=self.pca_period, n_components=3)
        # First component is column 0
        self._pca_first = pca_result[:, 0] if pca_result.ndim > 1 else pca_result
        self._oi_mom = oi_momentum(oi, period=self.oi_mom_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        pca_val = self._pca_first[i]
        oi_val = self._oi_mom[i]
        atr_val = self._atr[i]
        if np.isnan(pca_val) or np.isnan(oi_val) or np.isnan(atr_val):
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
        if side == 0 and pca_val > 0 and oi_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and pca_val < 0:
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
