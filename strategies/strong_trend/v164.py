"""
QBase Strong Trend Strategy v164 — LMS Filter (Adaptive) + McGinley Dynamic
=============================================================================

策略简介：最小均方自适应滤波器跟踪价格趋势，当LMS滤波输出上升且
         收盘价在McGinley动态均线上方时入场做多。

使用指标：
  - LMS Filter (period=20, mu=0.01): 自适应LMS滤波器
  - McGinley Dynamic (period=14): 自适应速度的动态均线
  - ATR (period=14): trailing stop

进场条件：
  1. LMS滤波值上升（当前 > 前一bar）
  2. 收盘价 > McGinley Dynamic

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. 收盘价 < McGinley Dynamic

优点：LMS自适应学习，无需手动调参
缺点：mu过大导致不稳定，过小则适应慢
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.ml.adaptive_lms import lms_filter
from indicators.trend.mcginley import mcginley_dynamic
from indicators.volatility.atr import atr


class StrongTrendV164(TimeSeriesStrategy):
    """LMS自适应滤波趋势 + McGinley动态均线。"""
    name = "strong_trend_v164"
    warmup = 60
    freq = "daily"

    lms_period: int = 20
    lms_mu: float = 0.01
    mcg_period: int = 14
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._lms = None
        self._mcg = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._lms = lms_filter(closes, period=self.lms_period, mu=self.lms_mu)
        self._mcg = mcginley_dynamic(closes, period=self.mcg_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        lms_val = self._lms[i]
        lms_prev = self._lms[i - 1] if i > 0 else np.nan
        mcg_val = self._mcg[i]
        atr_val = self._atr[i]
        if np.isnan(lms_val) or np.isnan(lms_prev) or np.isnan(mcg_val) or np.isnan(atr_val):
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
        if side == 0 and lms_val > lms_prev and price > mcg_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and price < mcg_val:
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
