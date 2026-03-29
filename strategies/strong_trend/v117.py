"""
Strong Trend Strategy v117 — Normalized Range Expansion + ROC
===============================================================
Uses normalized range (H-L)/C to detect volatility compression and
expansion, confirmed by Rate of Change for momentum direction.

  1. Normalized Range — (H-L)/C z-score for regime detection
  2. ROC             — momentum direction and strength

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v117.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.normalized_range import normalized_range
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr


class StrongTrendV117(TimeSeriesStrategy):
    """
    策略简介：标准化波幅扩张 + ROC动量确认的突破策略
    使用指标：Normalized Range（标准化波幅及Z-score）、ROC（变化率）
    进场条件：NR Z-score > 1.5 + ROC > 阈值
    出场条件：ATR trailing stop 或 ROC < 0
    优点：NR Z-score标准化，跨品种可比；ROC简单直接
    缺点：Z-score需要足够数据窗口，早期不稳定
    """
    name = "strong_trend_v117"
    warmup = 60
    freq = "daily"

    nr_period: int = 20
    roc_period: int = 12
    zscore_threshold: float = 1.5
    roc_threshold: float = 3.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._nr = None
        self._nr_zscore = None
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

        self._nr, self._nr_zscore = normalized_range(highs, lows, closes, period=self.nr_period)
        self._roc = rate_of_change(closes, self.roc_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        nr_z = self._nr_zscore[i]
        cur_roc = self._roc[i]
        atr_val = self._atr[i]

        if np.isnan(nr_z) or np.isnan(cur_roc) or np.isnan(atr_val):
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
        if side == 0 and nr_z > self.zscore_threshold and cur_roc > self.roc_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and cur_roc < 0:
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
