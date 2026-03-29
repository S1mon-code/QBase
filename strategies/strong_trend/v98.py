"""
QBase Strong Trend Strategy v98 — Changepoint Score + Fisher Transform
========================================================================

策略简介：Changepoint Score 检测结构性变化点（低分 = 当前 regime 稳定），
         Fisher Transform 确认价格动量方向。

使用指标：
  - Changepoint Score (period=60): 低值 = 无结构变化 = regime 稳定
  - Fisher Transform (period=10): 价格动量方向
  - ATR (period=14): trailing stop

进场条件：
  1. Changepoint Score < 0.3（regime 稳定）
  2. Fisher > trigger（动量向上）
  3. Fisher > 0（正动量区域）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Changepoint Score > 0.7（结构变化 = regime 切换）

优点：Changepoint 检测能提前预警 regime 切换
缺点：Fisher Transform 在极端值时可能失真
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.changepoint import changepoint_score
from indicators.momentum.fisher_transform import fisher_transform
from indicators.volatility.atr import atr


class StrongTrendV98(TimeSeriesStrategy):
    """Changepoint Score regime stability + Fisher Transform momentum."""
    name = "strong_trend_v98"
    warmup = 60
    freq = "daily"

    cp_stable_thresh: float = 0.3
    cp_change_thresh: float = 0.7
    fisher_period: int = 10
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._cp = None
        self._fisher = None
        self._trigger = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._cp = changepoint_score(closes, period=60)
        self._fisher, self._trigger = fisher_transform(highs, lows, period=self.fisher_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        cp_val = self._cp[i]
        fish = self._fisher[i]
        trig = self._trigger[i]
        atr_val = self._atr[i]
        if np.isnan(cp_val) or np.isnan(fish) or np.isnan(trig) or np.isnan(atr_val):
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
        if side == 0 and cp_val < self.cp_stable_thresh and fish > trig and fish > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and cp_val > self.cp_change_thresh:
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
