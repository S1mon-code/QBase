"""
QBase Strong Trend Strategy v160 — Momentum Decompose (Long Component) + Aroon
================================================================================

策略简介：动量分解提取长期动量分量，当长期动量为正且Aroon指标确认
         上升趋势时入场做多。

使用指标：
  - Momentum Decompose (short=5, medium=20, long=60): 动量分解
  - Aroon (period=25): 趋势方向与强度
  - ATR (period=14): trailing stop

进场条件：
  1. 长期动量分量 > 0（长周期趋势向上）
  2. Aroon Up > 70 且 Aroon Up > Aroon Down

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Aroon Down > Aroon Up（趋势翻转）

优点：长期动量分量过滤短期噪声
缺点：长周期分解信号滞后
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.ml.momentum_decompose import momentum_components
from indicators.trend.aroon import aroon
from indicators.volatility.atr import atr


class StrongTrendV160(TimeSeriesStrategy):
    """动量分解长期分量 + Aroon趋势方向确认。"""
    name = "strong_trend_v160"
    warmup = 60
    freq = "daily"

    mom_long: int = 60
    aroon_period: int = 25
    aroon_threshold: float = 70.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._mom_long = None
        self._aroon_up = None
        self._aroon_down = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        components = momentum_components(closes, short=5, medium=20, long=self.mom_long)
        # Long component is the last column
        if components.ndim > 1:
            self._mom_long = components[:, -1]
        else:
            self._mom_long = components
        self._aroon_up, self._aroon_down, _ = aroon(highs, lows, period=self.aroon_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        mom_val = self._mom_long[i]
        ar_up = self._aroon_up[i]
        ar_down = self._aroon_down[i]
        atr_val = self._atr[i]
        if np.isnan(mom_val) or np.isnan(ar_up) or np.isnan(ar_down) or np.isnan(atr_val):
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
        if side == 0 and mom_val > 0 and ar_up > self.aroon_threshold and ar_up > ar_down:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and ar_down > ar_up:
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
