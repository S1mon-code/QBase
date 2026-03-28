"""
QBase Strong Trend Strategy v85 — Trend Strength Composite + TEMA
==================================================================

策略简介：Trend Strength Composite 综合多维度评估趋势强度，
         TEMA 提供平滑的趋势方向确认。

使用指标：
  - Trend Strength Composite (period=20): 综合趋势强度评分
  - TEMA (period=20): 三重指数平滑均线
  - ATR (period=14): trailing stop

进场条件：
  1. Trend Strength > 0.6（强趋势评分）
  2. 收盘价 > TEMA（趋势方向向上）
  3. TEMA 斜率 > 0（趋势加速）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Trend Strength < 0.2（趋势衰弱）

优点：综合指标减少单一指标偏差，TEMA 反应快于 SMA/EMA
缺点：综合指标可能掩盖单维度的极端信号
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.trend_strength_composite import trend_strength
from indicators.trend.tema import tema
from indicators.volatility.atr import atr


class StrongTrendV85(TimeSeriesStrategy):
    """Trend Strength Composite regime + TEMA direction."""
    name = "strong_trend_v85"
    warmup = 60
    freq = "daily"

    ts_period: int = 20
    ts_threshold: float = 0.6
    tema_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._ts = None
        self._tema = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._ts = trend_strength(closes, highs, lows, period=self.ts_period)
        self._tema = tema(closes, period=self.tema_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        ts_val = self._ts[i]
        tema_val = self._tema[i]
        atr_val = self._atr[i]
        if np.isnan(ts_val) or np.isnan(tema_val) or np.isnan(atr_val) or i < 1:
            return

        tema_prev = self._tema[i - 1]
        if np.isnan(tema_prev):
            return

        tema_slope = tema_val - tema_prev

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
        if side == 0 and ts_val > self.ts_threshold and price > tema_val and tema_slope > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and ts_val < 0.2:
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
