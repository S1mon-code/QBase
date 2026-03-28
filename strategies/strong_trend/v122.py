"""
Strong Trend Strategy v122 — Close-to-Close Vol Expansion + ALMA
==================================================================
Detects close-to-close volatility expansion from low regimes, with
ALMA (Arnaud Legoux Moving Average) as a smooth trend direction filter.

  1. Close-to-Close Vol — classic volatility expansion detection
  2. ALMA              — Gaussian-weighted MA for trend direction

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v122.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.close_to_close_vol import close_to_close_vol
from indicators.trend.alma import alma
from indicators.volatility.atr import atr


class StrongTrendV122(TimeSeriesStrategy):
    """
    策略简介：Close-to-Close波动率扩张 + ALMA趋势过滤的突破策略
    使用指标：Close-to-Close Vol（收盘价波动率）、ALMA（高斯加权均线）
    进场条件：CTC Vol > 1.5倍均值 + 价格在ALMA上方 + ALMA上升
    出场条件：ATR trailing stop 或价格跌破ALMA
    优点：CTC Vol是最基础的波动率衡量，ALMA平滑度好
    缺点：CTC Vol效率低于OHLC方法，可能滞后
    """
    name = "strong_trend_v122"
    warmup = 60
    freq = "daily"

    ctc_period: int = 20
    alma_period: int = 20
    ctc_expansion: float = 1.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._ctc = None
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

        self._ctc = close_to_close_vol(closes, period=self.ctc_period)
        self._alma = alma(closes, period=self.alma_period)
        self._atr = atr(highs, lows, closes, period=14)

        # Rolling average of CTC vol
        n = len(closes)
        self._ctc_avg = np.full(n, np.nan)
        for k in range(39, n):
            window = self._ctc[k - 39:k + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                self._ctc_avg[k] = np.mean(valid)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        if i < 1:
            return

        cur_ctc = self._ctc[i]
        ctc_avg = self._ctc_avg[i]
        cur_alma = self._alma[i]
        prev_alma = self._alma[i - 1]
        atr_val = self._atr[i]

        if np.isnan(cur_ctc) or np.isnan(ctc_avg) or np.isnan(cur_alma) or np.isnan(prev_alma) or np.isnan(atr_val):
            return

        ctc_expanding = ctc_avg > 0 and cur_ctc > self.ctc_expansion * ctc_avg
        alma_rising = cur_alma > prev_alma
        above_alma = price > cur_alma

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
        if side == 0 and ctc_expanding and above_alma and alma_rising:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and price < cur_alma:
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
