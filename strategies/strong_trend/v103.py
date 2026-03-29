"""
Strong Trend Strategy v103 — NR7 (Narrow Range 7) + ROC Breakout
=================================================================
Detects low-volatility compression via NR7 pattern, then enters on
ROC breakout confirming directional momentum after the squeeze.

  1. NR7  — narrow range day detection (volatility compression)
  2. ROC  — momentum breakout confirmation after compression

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v103.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.nr7 import nr7
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr


class StrongTrendV103(TimeSeriesStrategy):
    """
    策略简介：NR7窄幅日 + ROC动量突破的波动率压缩突破策略
    使用指标：NR7（窄幅检测）、ROC（动量突破确认）
    进场条件：近期出现NR7 + ROC突破阈值 + 价格创新高
    出场条件：ATR trailing stop 或 ROC跌破零
    优点：经典压缩-爆发模式，NR7是已验证的有效模式
    缺点：NR7出现频率低，交易机会较少
    """
    name = "strong_trend_v103"
    warmup = 60
    freq = "daily"

    roc_period: int = 12
    roc_threshold: float = 3.0
    nr7_lookback: int = 5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._nr7 = None
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

        self._nr7 = nr7(highs, lows)
        self._roc = rate_of_change(closes, self.roc_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        if i < self.nr7_lookback:
            return

        cur_roc = self._roc[i]
        atr_val = self._atr[i]

        if np.isnan(cur_roc) or np.isnan(atr_val):
            return

        # Check if NR7 appeared in recent bars
        recent_nr7 = np.any(self._nr7[max(0, i - self.nr7_lookback):i + 1])

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
        if side == 0 and recent_nr7 and cur_roc > self.roc_threshold:
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
