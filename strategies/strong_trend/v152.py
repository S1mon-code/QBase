"""
QBase Strong Trend Strategy v152 — K-Means Regime (Trending Cluster) + ROC
============================================================================

策略简介：使用 K-Means 聚类识别市场 regime，当检测到趋势 regime 且
         ROC 动量为正时入场做多。

使用指标：
  - K-Means Regime (n_clusters=3): 基于特征矩阵的 regime 聚类
  - Rate of Change (period=12): 动量确认
  - ATR (period=14): trailing stop

进场条件：
  1. K-Means regime 标识为趋势状态（regime == 趋势 cluster）
  2. ROC > 0（正向动量）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. ROC 转负

优点：数据驱动的 regime 识别，无需主观判断
缺点：cluster 标签不稳定，需要足够数据
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
from indicators.ml.kmeans_regime import kmeans_regime
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr


class StrongTrendV152(TimeSeriesStrategy):
    """K-Means regime聚类趋势识别 + ROC动量确认。"""
    name = "strong_trend_v152"
    warmup = 60
    freq = "daily"

    kmeans_period: int = 120
    roc_period: int = 12
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._regime = None
        self._roc = None
        self._atr = None
        self._trending_cluster = None

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

        self._regime = kmeans_regime(features, period=self.kmeans_period, n_clusters=3)
        self._roc = rate_of_change(closes, period=self.roc_period)
        self._atr = atr(highs, lows, closes, period=14)

        # Identify trending cluster: the one with highest mean ADX
        valid_mask = ~np.isnan(self._regime) & ~np.isnan(adx_arr)
        if np.any(valid_mask):
            cluster_adx = {}
            for c in range(3):
                mask = valid_mask & (self._regime == c)
                if np.any(mask):
                    cluster_adx[c] = np.nanmean(adx_arr[mask])
            if cluster_adx:
                self._trending_cluster = max(cluster_adx, key=cluster_adx.get)
            else:
                self._trending_cluster = 0
        else:
            self._trending_cluster = 0

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        regime_val = self._regime[i]
        roc_val = self._roc[i]
        atr_val = self._atr[i]
        if np.isnan(regime_val) or np.isnan(roc_val) or np.isnan(atr_val):
            return

        is_trending = int(regime_val) == self._trending_cluster

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
        if side == 0 and is_trending and roc_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and roc_val < 0:
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
