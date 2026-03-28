"""
QBase Strong Trend Strategy v190 — Schaff Trend Cycle + Supertrend
====================================================================

策略简介：Schaff Trend Cycle将MACD通过双重Stochastic平滑，产生0-100范围的趋势信号。
         结合Supertrend方向过滤，在强趋势启动时入场。

使用指标：
  - Schaff Trend Cycle (period=10, fast=23, slow=50): 趋势周期指标
  - Supertrend (period=10, mult=3.0): 趋势方向
  - ATR (period=14): trailing stop

进场条件：
  1. STC > 75（强上升趋势区）
  2. Supertrend方向为多头（dir == 1）

出场条件：
  1. ATR trailing stop (mult=4.0)
  2. STC < 25（进入下降趋势区）

优点：STC响应快 + Supertrend方向确认
缺点：STC在震荡市可能频繁触发
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr


class StrongTrendV190(TimeSeriesStrategy):
    """Schaff趋势周期 + Supertrend方向策略。"""
    name = "strong_trend_v190"
    warmup = 60
    freq = "daily"

    stc_period: int = 10
    st_period: int = 10
    st_mult: float = 3.0
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._stc = None
        self._st_dir = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._stc = schaff_trend_cycle(closes, period=self.stc_period, fast=23, slow=50)
        _, self._st_dir = supertrend(highs, lows, closes,
                                      period=self.st_period, mult=self.st_mult)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        stc_val = self._stc[i]
        st_d = self._st_dir[i]
        atr_val = self._atr[i]
        if np.isnan(stc_val) or np.isnan(st_d) or np.isnan(atr_val):
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
        if side == 0 and stc_val > 75.0 and st_d == 1:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and stc_val < 25.0:
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
