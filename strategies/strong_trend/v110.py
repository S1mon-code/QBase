"""
Strong Trend Strategy v110 — Parkinson Volatility + McGinley Dynamic
=====================================================================
Uses Parkinson high-low volatility estimator for expansion detection,
with McGinley Dynamic as an adaptive trend filter.

  1. Parkinson Vol     — high-low range volatility (5.2x efficiency)
  2. McGinley Dynamic  — adaptive moving average for trend direction

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v110.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.parkinson import parkinson
from indicators.trend.mcginley import mcginley_dynamic
from indicators.volatility.atr import atr


class StrongTrendV110(TimeSeriesStrategy):
    """
    策略简介：Parkinson波动率扩张 + McGinley Dynamic趋势过滤的突破策略
    使用指标：Parkinson Vol（高低价波动率）、McGinley Dynamic（自适应均线）
    进场条件：Parkinson Vol扩张 + 价格在McGinley上方 + McGinley上升
    出场条件：ATR trailing stop 或价格跌破McGinley
    优点：Parkinson仅用高低价，不受收盘价噪音影响；McGinley自适应调速
    缺点：Parkinson假设无漂移，强趋势中可能低估
    """
    name = "strong_trend_v110"
    warmup = 60
    freq = "daily"

    park_period: int = 20
    mg_period: int = 14
    park_expansion: float = 1.4
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._park = None
        self._mg = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._park = parkinson(highs, lows, period=self.park_period)
        self._mg = mcginley_dynamic(closes, period=self.mg_period)
        self._atr = atr(highs, lows, closes, period=14)

        # Rolling average of Parkinson vol
        n = len(closes)
        self._park_avg = np.full(n, np.nan)
        for k in range(39, n):
            window = self._park[k - 39:k + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                self._park_avg[k] = np.mean(valid)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        if i < 1:
            return

        cur_park = self._park[i]
        park_avg = self._park_avg[i]
        cur_mg = self._mg[i]
        prev_mg = self._mg[i - 1]
        atr_val = self._atr[i]

        if np.isnan(cur_park) or np.isnan(park_avg) or np.isnan(cur_mg) or np.isnan(prev_mg) or np.isnan(atr_val):
            return

        park_expanding = park_avg > 0 and cur_park > self.park_expansion * park_avg
        mg_rising = cur_mg > prev_mg
        above_mg = price > cur_mg

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
        if side == 0 and park_expanding and above_mg and mg_rising:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and price < cur_mg:
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
