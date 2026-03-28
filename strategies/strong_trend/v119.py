"""
Strong Trend Strategy v119 — Relative Volatility (Fast vs Slow) + Supertrend
==============================================================================
Detects volatility expansion using fast/slow volatility ratio, confirmed
by Supertrend for trend direction and dynamic exit.

  1. Relative Volatility — fast_vol / slow_vol ratio and z-score
  2. Supertrend          — trend direction filter and trailing stop

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v119.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.relative_vol import relative_volatility
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr


class StrongTrendV119(TimeSeriesStrategy):
    """
    策略简介：相对波动率扩张 + Supertrend趋势确认的突破策略
    使用指标：Relative Volatility（快/慢波动率比）、Supertrend（趋势方向）
    进场条件：RV > 1 (expanding) + RV Z-score > 1 + Supertrend看多
    出场条件：ATR trailing stop 或 Supertrend翻空
    优点：相对波动率标准化，Z-score提供统计显著性；Supertrend方向明确
    缺点：波动率扩张不一定伴随上涨趋势
    """
    name = "strong_trend_v119"
    warmup = 60
    freq = "daily"

    rv_fast: int = 10
    rv_slow: int = 60
    st_period: int = 10
    st_mult: float = 3.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._rv = None
        self._rv_zscore = None
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

        self._rv, self._rv_zscore = relative_volatility(closes, fast=self.rv_fast, slow=self.rv_slow)
        self._st_line, self._st_dir = supertrend(highs, lows, closes, self.st_period, self.st_mult)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        rv = self._rv[i]
        rv_z = self._rv_zscore[i]
        st_dir = self._st_dir[i]
        atr_val = self._atr[i]

        if np.isnan(rv) or np.isnan(rv_z) or np.isnan(st_dir) or np.isnan(atr_val):
            return

        vol_expanding = rv > 1.0 and rv_z > 1.0
        trend_bullish = st_dir == 1

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
        if side == 0 and vol_expanding and trend_bullish:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and st_dir == -1:
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
