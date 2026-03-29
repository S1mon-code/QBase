"""
QBase Strong Trend Strategy v194 — Relative Vigor Index + EMA Ribbon
======================================================================

策略简介：RVI衡量收盘价相对于开盘到收盘范围的位置（类似力量指标），
         结合EMA Ribbon判断趋势是否展开。

使用指标：
  - Relative Vigor Index (period=10): 活力指数
  - EMA Ribbon Signal: EMA带状信号（多条EMA排列）
  - ATR (period=14): trailing stop

进场条件：
  1. RVI > RVI Signal（活力上穿信号线）
  2. EMA Ribbon Signal > 0.5（EMA多头排列）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. EMA Ribbon Signal < -0.5（EMA空头排列）

优点：RVI量化动能 + EMA Ribbon趋势确认
缺点：RVI信号频繁，需要EMA过滤
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.momentum.relative_vigor import relative_vigor_index
from indicators.trend.ema_ribbon import ema_ribbon_signal
from indicators.volatility.atr import atr


class StrongTrendV194(TimeSeriesStrategy):
    """相对活力指数 + EMA Ribbon策略。"""
    name = "strong_trend_v194"
    warmup = 60
    freq = "daily"

    rvi_period: int = 10
    ribbon_periods: list = None
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        if self.ribbon_periods is None:
            self.ribbon_periods = [10, 20, 30, 40, 50]
        self._rvi = None
        self._rvi_sig = None
        self._ribbon = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        opens = context.get_full_open_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._rvi, self._rvi_sig = relative_vigor_index(opens, highs, lows, closes,
                                                         period=self.rvi_period)
        self._ribbon = ema_ribbon_signal(closes, self.ribbon_periods)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        rvi_val = self._rvi[i]
        rvi_s = self._rvi_sig[i]
        rib = self._ribbon[i]
        atr_val = self._atr[i]
        if np.isnan(rvi_val) or np.isnan(rvi_s) or np.isnan(rib) or np.isnan(atr_val):
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
        if side == 0 and rvi_val > rvi_s and rib > 0.5:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and rib < -0.5:
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
