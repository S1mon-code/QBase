"""
QBase Strong Trend Strategy v80 — Volatility Clustering + Aroon
================================================================

策略简介：利用 Volatility Clustering（波动率聚集）检测 regime，
         高聚集意味着趋势市场，配合 Aroon 指标确认方向。

使用指标：
  - Volatility Clustering (period=60): 高值表示波动率有聚集性（趋势市场特征）
  - Aroon (period=25): Aroon Up > 70 确认上升趋势
  - ATR (period=14): trailing stop

进场条件：
  1. Vol Clustering > 0.3（波动率聚集 regime）
  2. Aroon Up > 70 且 Aroon Down < 30（强上升趋势）

出场条件：
  1. ATR trailing stop（mult=4.0）
  2. Aroon Down > 70（趋势反转信号）

优点：波动率聚集是趋势市场的统计特征，理论基础扎实
缺点：波动率聚集在突发事件后也会出现，需要方向过滤
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.volatility_clustering import vol_clustering
from indicators.trend.aroon import aroon
from indicators.volatility.atr import atr


class StrongTrendV80(TimeSeriesStrategy):
    """Volatility Clustering regime + Aroon trend direction."""
    name = "strong_trend_v80"
    warmup = 60
    freq = "daily"

    cluster_threshold: float = 0.3
    aroon_period: int = 25
    aroon_up_thresh: float = 70.0
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._vol_cluster = None
        self._aroon_up = None
        self._aroon_down = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._vol_cluster = vol_clustering(closes, period=60)
        self._aroon_up, self._aroon_down, _ = aroon(highs, lows, period=self.aroon_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        vc_val = self._vol_cluster[i]
        ar_up = self._aroon_up[i]
        ar_down = self._aroon_down[i]
        atr_val = self._atr[i]
        if np.isnan(vc_val) or np.isnan(ar_up) or np.isnan(ar_down) or np.isnan(atr_val):
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
        if side == 0 and vc_val > self.cluster_threshold and ar_up > self.aroon_up_thresh and ar_down < 30.0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and ar_down > 70.0:
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
