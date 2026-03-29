"""
Strong Trend Strategy v107 — Range Expansion Index + TEMA
==========================================================
Uses Range Expansion Index to detect bars with above-average range
(volatility breakout), confirmed by TEMA for trend direction.

  1. Range Expansion — current range vs average range ratio
  2. TEMA            — low-lag trend filter and direction

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v107.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.range_expansion import range_expansion
from indicators.trend.tema import tema
from indicators.volatility.atr import atr


class StrongTrendV107(TimeSeriesStrategy):
    """
    策略简介：Range Expansion + TEMA趋势过滤的波动率突破策略
    使用指标：Range Expansion Index（波幅扩张比）、TEMA（趋势方向）
    进场条件：REI > 1.5 + 价格在TEMA上方 + TEMA上升
    出场条件：ATR trailing stop 或价格跌破TEMA
    优点：REI直观，TEMA低延迟过滤方向
    缺点：单根大阳线也可能触发，需TEMA配合过滤
    """
    name = "strong_trend_v107"
    warmup = 60
    freq = "daily"

    rei_period: int = 14
    tema_period: int = 20
    rei_threshold: float = 1.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._rei = None
        self._tema = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._rei = range_expansion(highs, lows, closes, self.rei_period)
        self._tema = tema(closes, self.tema_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        if i < 1:
            return

        rei_val = self._rei[i]
        cur_tema = self._tema[i]
        prev_tema = self._tema[i - 1]
        atr_val = self._atr[i]

        if np.isnan(rei_val) or np.isnan(cur_tema) or np.isnan(prev_tema) or np.isnan(atr_val):
            return

        tema_rising = cur_tema > prev_tema
        above_tema = price > cur_tema

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
        if side == 0 and rei_val > self.rei_threshold and above_tema and tema_rising:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and price < cur_tema:
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
