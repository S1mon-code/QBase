"""
QBase Strong Trend Strategy v177 — Weekday Effect + ADX Filter
================================================================

策略简介：利用星期效应识别有利的交易日，结合ADX过滤确保在趋势充分的环境中入场。

使用指标：
  - Weekday Effect (lookback=252): 星期效应Z分数
  - ADX (period=14): 趋势强度过滤
  - ATR (period=14): trailing stop

进场条件：
  1. 星期效应Z分数 > 0.5（当日为历史看涨日）
  2. ADX > 20（趋势足够强）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. ADX < 15（趋势消散）

优点：利用日历效应 + 趋势过滤双重确认
缺点：星期效应可能随时间衰减
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.seasonality.weekday_effect import weekday_effect
from indicators.trend.adx import adx
from indicators.volatility.atr import atr


class StrongTrendV177(TimeSeriesStrategy):
    """星期效应 + ADX趋势过滤策略。"""
    name = "strong_trend_v177"
    warmup = 60
    freq = "daily"

    weekday_lookback: int = 252
    adx_period: int = 14
    adx_entry: float = 20.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._weekday_score = None
        self._adx = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        datetimes = context.get_full_datetime_array()

        self._weekday_score, _ = weekday_effect(closes, datetimes, lookback=self.weekday_lookback)
        self._adx = adx(highs, lows, closes, period=self.adx_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        wd = self._weekday_score[i]
        adx_val = self._adx[i]
        atr_val = self._atr[i]
        if np.isnan(wd) or np.isnan(adx_val) or np.isnan(atr_val):
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
        if side == 0 and wd > 0.5 and adx_val > self.adx_entry:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and adx_val < 15.0:
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
