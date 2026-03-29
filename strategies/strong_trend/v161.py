"""
QBase Strong Trend Strategy v161 — Decision Boundary Distance + TEMA
=====================================================================

策略简介：计算当前市场状态到分类决策边界的距离，距离越远表示趋势信号
         越强，配合TEMA确认方向后入场。

使用指标：
  - Decision Boundary Distance (period=120): 到分类边界的距离
  - TEMA (period=20): 三重指数移动平均趋势方向
  - ATR (period=14): trailing stop

进场条件：
  1. Decision boundary distance > 0（正类一侧，即趋势方向）
  2. 收盘价 > TEMA（价格在趋势线上方）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Decision boundary distance < 0（穿越到负类一侧）

优点：距离大小反映信号置信度
缺点：需要标签构建，依赖特征质量
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
from indicators.ml.decision_boundary import decision_boundary_distance
from indicators.trend.tema import tema
from indicators.volatility.atr import atr


class StrongTrendV161(TimeSeriesStrategy):
    """决策边界距离信号强度 + TEMA趋势方向。"""
    name = "strong_trend_v161"
    warmup = 60
    freq = "daily"

    db_period: int = 120
    tema_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._db_dist = None
        self._tema = None
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

        # Create labels: 1 if price went up over next 5 bars, else 0
        returns = np.full(len(closes), np.nan)
        returns[:-5] = (closes[5:] - closes[:-5]) / closes[:-5]
        labels = (returns > 0).astype(float)
        labels[np.isnan(returns)] = np.nan

        self._db_dist = decision_boundary_distance(features, labels, period=self.db_period)
        self._tema = tema(closes, period=self.tema_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        dist_val = self._db_dist[i]
        tema_val = self._tema[i]
        atr_val = self._atr[i]
        if np.isnan(dist_val) or np.isnan(tema_val) or np.isnan(atr_val):
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
        if side == 0 and dist_val > 0 and price > tema_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and dist_val < 0:
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
