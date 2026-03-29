"""
QBase Strong Trend Strategy v155 — Bayesian Online Trend + Force Index
=======================================================================

策略简介：贝叶斯在线趋势检测实时估计趋势概率，当趋势概率高且
         Force Index 确认买压时入场做多。

使用指标：
  - Bayesian Online Trend (hazard_rate=0.01): 在线趋势概率估计
  - Force Index (period=13): 量价合力指标
  - ATR (period=14): trailing stop

进场条件：
  1. Bayesian trend > 0.5（趋势概率超过50%）
  2. Force Index > 0（买压大于卖压）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Bayesian trend < 0.3（趋势概率降低）

优点：贝叶斯方法自适应更新，对 regime 变化敏感
缺点：hazard rate 选择影响变点检测灵敏度
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.ml.bayesian_trend import bayesian_online_trend
from indicators.volume.force_index import force_index
from indicators.volatility.atr import atr


class StrongTrendV155(TimeSeriesStrategy):
    """贝叶斯在线趋势检测 + Force Index量价确认。"""
    name = "strong_trend_v155"
    warmup = 60
    freq = "daily"

    hazard_rate: float = 0.01
    force_period: int = 13
    trend_threshold: float = 0.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._bayesian = None
        self._force = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._bayesian = bayesian_online_trend(closes, hazard_rate=self.hazard_rate)
        self._force = force_index(closes, volumes, period=self.force_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        bayes_val = self._bayesian[i]
        force_val = self._force[i]
        atr_val = self._atr[i]
        if np.isnan(bayes_val) or np.isnan(force_val) or np.isnan(atr_val):
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
        if side == 0 and bayes_val > self.trend_threshold and force_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and bayes_val < 0.3:
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
