"""
QBase Strong Trend Strategy v100 — CUSUM Structural Break + Supertrend
========================================================================

策略简介：CUSUM 检测结构性突变点，在无突变的稳定趋势 regime 中，
         配合 Supertrend 确认方向后入场。

使用指标：
  - CUSUM Structural Break (period=60, threshold=3.0): 结构突变检测
  - Supertrend (period=10, mult=3.0): 趋势方向与动态支撑
  - ATR (period=14): trailing stop

进场条件：
  1. CUSUM break 未触发（cusum < threshold, regime 稳定）
  2. Supertrend direction == 1（上升趋势）
  3. 收盘价 > Supertrend line

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Supertrend direction 翻转为 -1
  3. CUSUM break 触发（结构突变 = 紧急退出）

优点：CUSUM 是经典质量控制方法，能可靠检测均值偏移
缺点：阈值选择影响灵敏度，过低则频繁触发
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.structural_break import cusum_break
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr


class StrongTrendV100(TimeSeriesStrategy):
    """CUSUM structural break regime + Supertrend direction."""
    name = "strong_trend_v100"
    warmup = 60
    freq = "daily"

    cusum_period: int = 60
    cusum_threshold: float = 3.0
    st_period: int = 10
    st_mult: float = 3.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._cusum = None
        self._st_line = None
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

        self._cusum = cusum_break(closes, period=self.cusum_period, threshold=self.cusum_threshold)
        self._st_line, self._st_dir = supertrend(highs, lows, closes, period=self.st_period, mult=self.st_mult)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        cusum_val = self._cusum[i]
        st_dir = self._st_dir[i]
        st_line = self._st_line[i]
        atr_val = self._atr[i]
        if np.isnan(cusum_val) or np.isnan(st_dir) or np.isnan(st_line) or np.isnan(atr_val):
            return

        # CUSUM < threshold means no structural break detected = stable regime
        stable = cusum_val < self.cusum_threshold

        # === Stop Loss Check ===
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # === Emergency Exit on Structural Break ===
        if side == 1 and not stable:
            context.close_long()
            self._reset()
            return

        # === Entry ===
        if side == 0 and stable and st_dir == 1 and price > st_line:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and st_dir == -1:
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
