"""
QBase Strong Trend Strategy v193 — Pretty Good Oscillator + HMA
==================================================================

策略简介：Pretty Good Oscillator (PGO)衡量价格相对ATR的偏离度，结合HMA趋势确认。
         PGO > 2表示价格大幅高于近期范围（突破信号）。

使用指标：
  - Pretty Good Oscillator (period=14): 价格偏离度
  - HMA (period=20): Hull移动平均趋势
  - ATR (period=14): trailing stop

进场条件：
  1. PGO > 2.0（价格显著偏离向上）
  2. 价格 > HMA（Hull均线确认趋势）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. PGO < 0（价格回到平均以下）

优点：PGO简单有效衡量突破强度，HMA低延迟
缺点：PGO阈值需要品种适配
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.momentum.pretty_good_oscillator import pretty_good_oscillator
from indicators.trend.hma import hma
from indicators.volatility.atr import atr


class StrongTrendV193(TimeSeriesStrategy):
    """Pretty Good Oscillator突破 + HMA趋势策略。"""
    name = "strong_trend_v193"
    warmup = 60
    freq = "daily"

    pgo_period: int = 14
    hma_period: int = 20
    pgo_entry: float = 2.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._pgo = None
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

        self._pgo = pretty_good_oscillator(closes, highs, lows, period=self.pgo_period)
        self._hma = hma(closes, period=self.hma_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        pgo_val = self._pgo[i]
        hma_val = self._hma[i]
        atr_val = self._atr[i]
        if np.isnan(pgo_val) or np.isnan(hma_val) or np.isnan(atr_val):
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
        if side == 0 and pgo_val > self.pgo_entry and price > hma_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and pgo_val < 0.0:
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
