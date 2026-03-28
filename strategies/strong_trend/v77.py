"""
QBase Strong Trend Strategy v77 — Efficiency Ratio + Supertrend
================================================================

策略简介：用 Efficiency Ratio 识别高效趋势 regime，Supertrend 指标确认
         趋势方向后入场做多。

使用指标：
  - Efficiency Ratio (period=20): ER > 0.4 表示市场运动高效（强趋势）
  - Supertrend (period=10, mult=3.0): 方向确认与动态支撑
  - ATR (period=14): trailing stop

进场条件：
  1. Efficiency Ratio > 0.4（趋势高效）
  2. Supertrend direction == 1（上升趋势）
  3. 收盘价 > Supertrend line

出场条件：
  1. ATR trailing stop（mult=4.0）
  2. Supertrend direction 翻转为 -1

优点：ER 是纯数学测量，不依赖参数假设；Supertrend 提供清晰方向信号
缺点：ER 在窄幅振荡中可能给出误导信号
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.efficiency_ratio import efficiency_ratio
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr


class StrongTrendV77(TimeSeriesStrategy):
    """Efficiency Ratio regime filter + Supertrend direction."""
    name = "strong_trend_v77"
    warmup = 60
    freq = "daily"

    er_period: int = 20
    er_threshold: float = 0.4
    st_period: int = 10
    st_mult: float = 3.0
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._er = None
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

        self._er = efficiency_ratio(closes, period=self.er_period)
        self._st_line, self._st_dir = supertrend(highs, lows, closes, period=self.st_period, mult=self.st_mult)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        er_val = self._er[i]
        st_dir = self._st_dir[i]
        st_line = self._st_line[i]
        atr_val = self._atr[i]
        if np.isnan(er_val) or np.isnan(st_dir) or np.isnan(atr_val):
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
        if side == 0 and er_val > self.er_threshold and st_dir == 1 and price > st_line:
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
