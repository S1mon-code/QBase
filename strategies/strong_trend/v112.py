"""
Strong Trend Strategy v112 — Realized Variance Spike + HMA
============================================================
Detects realized variance spikes indicating regime change, with Hull
Moving Average providing a low-lag trend direction filter.

  1. Realized Variance — sum of squared returns spike detection
  2. HMA              — ultra-low-lag trend filter

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v112.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.realized_variance import realized_variance
from indicators.trend.hma import hma
from indicators.volatility.atr import atr


class StrongTrendV112(TimeSeriesStrategy):
    """
    策略简介：实现方差突增 + HMA低延迟趋势过滤的突破策略
    使用指标：Realized Variance（实现方差）、HMA（Hull均线）
    进场条件：RV > 2倍均值 + HMA上升 + 价格在HMA上方
    出场条件：ATR trailing stop 或 HMA转跌
    优点：RV直接衡量波动强度，HMA延迟极低
    缺点：RV可能在暴跌时也突增，需HMA方向过滤
    """
    name = "strong_trend_v112"
    warmup = 60
    freq = "daily"

    rv_period: int = 20
    hma_period: int = 20
    rv_expansion: float = 2.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._rv = None
        self._hma = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._rv = realized_variance(closes, period=self.rv_period)
        self._hma = hma(closes, period=self.hma_period)
        self._atr = atr(highs, lows, closes, period=14)

        # Rolling average of RV
        n = len(closes)
        self._rv_avg = np.full(n, np.nan)
        for k in range(39, n):
            window = self._rv[k - 39:k + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                self._rv_avg[k] = np.mean(valid)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        if i < 1:
            return

        cur_rv = self._rv[i]
        rv_avg = self._rv_avg[i]
        cur_hma = self._hma[i]
        prev_hma = self._hma[i - 1]
        atr_val = self._atr[i]

        if np.isnan(cur_rv) or np.isnan(rv_avg) or np.isnan(cur_hma) or np.isnan(prev_hma) or np.isnan(atr_val):
            return

        rv_spiking = rv_avg > 0 and cur_rv > self.rv_expansion * rv_avg
        hma_rising = cur_hma > prev_hma
        above_hma = price > cur_hma

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
        if side == 0 and rv_spiking and hma_rising and above_hma:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and not hma_rising:
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
