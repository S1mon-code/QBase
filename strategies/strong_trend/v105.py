"""
Strong Trend Strategy v105 — Historical Volatility Expansion + Supertrend
==========================================================================
Detects volatility regime expansion using historical volatility, confirmed
by Supertrend for trend direction and trailing stop.

  1. Historical Volatility — regime shift detection (expansion from low)
  2. Supertrend            — directional filter and exit mechanism

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v105.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.historical_vol import historical_volatility
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr


class StrongTrendV105(TimeSeriesStrategy):
    """
    策略简介：历史波动率扩张 + Supertrend趋势方向的波动率突破策略
    使用指标：Historical Volatility（波动率状态）、Supertrend（趋势方向）
    进场条件：HV从低位扩张超过均值 + Supertrend看多
    出场条件：ATR trailing stop 或 Supertrend翻空
    优点：HV扩张是趋势启动的可靠信号，Supertrend提供清晰方向
    缺点：HV有滞后性，扩张确认时可能已错过部分行情
    """
    name = "strong_trend_v105"
    warmup = 60
    freq = "daily"

    hv_period: int = 20
    st_period: int = 10
    st_mult: float = 3.0
    atr_trail_mult: float = 4.5
    hv_expansion: float = 1.5

    def __init__(self):
        super().__init__()
        self._hv = None
        self._st_line = None
        self._st_dir = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._hv = historical_volatility(closes, self.hv_period)
        self._st_line, self._st_dir = supertrend(highs, lows, closes, self.st_period, self.st_mult)
        self._atr = atr(highs, lows, closes, period=14)

        # Pre-compute rolling mean of HV
        n = len(closes)
        self._hv_avg = np.full(n, np.nan)
        for k in range(39, n):
            window = self._hv[k - 39:k + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                self._hv_avg[k] = np.mean(valid)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        cur_hv = self._hv[i]
        hv_avg = self._hv_avg[i]
        cur_dir = self._st_dir[i]
        atr_val = self._atr[i]

        if np.isnan(cur_hv) or np.isnan(hv_avg) or np.isnan(cur_dir) or np.isnan(atr_val):
            return

        hv_expanding = cur_hv > self.hv_expansion * hv_avg

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
        if side == 0 and hv_expanding and cur_dir == 1:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and cur_dir == -1:
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
