"""
QBase Strong Trend Strategy v93 — Adaptive Lookback + ALMA
============================================================

策略简介：Adaptive Lookback 动态调整回溯周期（短周期 = 快速趋势），
         ALMA 作为低延迟均线确认方向。

使用指标：
  - Adaptive Lookback (min_period=10, max_period=100): 自适应周期
  - ALMA (period=9, offset=0.85, sigma=6): Arnaud Legoux MA
  - ATR (period=14): trailing stop

进场条件：
  1. Adaptive Lookback < 30（短周期 = 快速趋势 regime）
  2. 收盘价 > ALMA（趋势向上）
  3. 收盘价 > 前一日收盘价

出场条件：
  1. ATR trailing stop（mult=4.0）
  2. Adaptive Lookback > 70（长周期 = 趋势放缓）

优点：自适应周期能动态捕捉趋势速度变化
缺点：周期估计在转折点附近不稳定
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.adaptive_period import adaptive_lookback
from indicators.trend.alma import alma
from indicators.volatility.atr import atr


class StrongTrendV93(TimeSeriesStrategy):
    """Adaptive Lookback regime + ALMA direction."""
    name = "strong_trend_v93"
    warmup = 60
    freq = "daily"

    al_short_thresh: float = 30.0
    alma_period: int = 9
    alma_offset: float = 0.85
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._al = None
        self._alma = None
        self._atr = None
        self._closes = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._al = adaptive_lookback(closes, min_period=10, max_period=100)
        self._alma = alma(closes, period=self.alma_period, offset=self.alma_offset, sigma=6)
        self._atr = atr(highs, lows, closes, period=14)
        self._closes = closes

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        al_val = self._al[i]
        alma_val = self._alma[i]
        atr_val = self._atr[i]
        if np.isnan(al_val) or np.isnan(alma_val) or np.isnan(atr_val) or i < 1:
            return

        prev_close = self._closes[i - 1]

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
        if side == 0 and al_val < self.al_short_thresh and price > alma_val and price > prev_close:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and al_val > 70.0:
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
