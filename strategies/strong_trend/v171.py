"""
QBase Strong Trend Strategy v171 — Information Ratio + Supertrend
==================================================================

策略简介：滚动信息比率衡量策略超额收益的稳定性，当IR较高说明
         趋势收益稳定，配合Supertrend确认方向后入场。

使用指标：
  - Information Ratio (period=60): 超额收益/跟踪误差
  - Supertrend (period=10, mult=3.0): 趋势方向与动态支撑
  - ATR (period=14): trailing stop

进场条件：
  1. Information Ratio > 0.5（超额收益稳定）
  2. Supertrend direction == 1（上升趋势）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Supertrend direction 翻转为 -1

优点：IR过滤不稳定的趋势信号
缺点：需要合理的benchmark构建
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.ml.information_ratio import rolling_information_ratio
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr


class StrongTrendV171(TimeSeriesStrategy):
    """信息比率趋势稳定性 + Supertrend方向。"""
    name = "strong_trend_v171"
    warmup = 60
    freq = "daily"

    ir_period: int = 60
    st_period: int = 10
    st_mult: float = 3.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._ir = None
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

        # Returns as strategy returns; benchmark = buy-and-hold (SMA proxy)
        returns = np.full(len(closes), np.nan)
        returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
        # Benchmark: rolling mean return
        benchmark = np.full(len(closes), np.nan)
        for j in range(20, len(closes)):
            benchmark[j] = np.nanmean(returns[j - 20:j])

        self._ir = rolling_information_ratio(returns, benchmark, period=self.ir_period)
        self._st_line, self._st_dir = supertrend(highs, lows, closes, period=self.st_period, mult=self.st_mult)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        ir_val = self._ir[i]
        st_dir = self._st_dir[i]
        atr_val = self._atr[i]
        if np.isnan(ir_val) or np.isnan(st_dir) or np.isnan(atr_val):
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
        if side == 0 and ir_val > 0.5 and st_dir == 1:
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
