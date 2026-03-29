"""
QBase Strong Trend Strategy v199 — Sine Wave (Ehlers) + ALMA
===============================================================

策略简介：Ehlers Sine Wave指标检测市场周期状态，当sine上穿lead sine（周期底部转折）
         且ALMA确认趋势方向时做多。

使用指标：
  - Ehlers Sine Wave (alpha=0.07): 周期分析（sine, lead_sine）
  - ALMA (period=9, offset=0.85, sigma=6): Arnaud Legoux移动平均
  - ATR (period=14): trailing stop

进场条件：
  1. sine > lead_sine（周期上升阶段）
  2. 价格 > ALMA（趋势向上）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. sine < lead_sine（周期下降阶段）

优点：Ehlers周期分析精确，ALMA低延迟
缺点：周期分析在非周期行情中失效
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.trend.sine_wave import ehlers_sine_wave
from indicators.trend.alma import alma
from indicators.volatility.atr import atr


class StrongTrendV199(TimeSeriesStrategy):
    """Ehlers Sine Wave周期 + ALMA趋势策略。"""
    name = "strong_trend_v199"
    warmup = 60
    freq = "daily"

    sw_alpha: float = 0.07
    alma_period: int = 9
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._sine = None
        self._lead = None
        self._alma = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        sw_result = ehlers_sine_wave(closes, alpha=self.sw_alpha)
        # Returns (sine, lead_sine) or similar tuple
        if isinstance(sw_result, tuple) and len(sw_result) >= 2:
            self._sine = sw_result[0]
            self._lead = sw_result[1]
        else:
            self._sine = sw_result
            self._lead = np.roll(sw_result, 1)
            self._lead[0] = np.nan

        self._alma = alma(closes, period=self.alma_period, offset=0.85, sigma=6)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        s = self._sine[i]
        l = self._lead[i]
        alma_val = self._alma[i]
        atr_val = self._atr[i]
        if np.isnan(s) or np.isnan(l) or np.isnan(alma_val) or np.isnan(atr_val):
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
        if side == 0 and s > l and price > alma_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and s < l:
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
