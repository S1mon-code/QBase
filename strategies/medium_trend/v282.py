import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.seasonality.month_turn_effect import month_turn
from indicators.momentum.macd import macd
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV282(TimeSeriesStrategy):
    """
    策略简介：月末月初效应+MACD趋势确认。
    使用指标：Month Turn Effect(3) + MACD(12,26,9) + ATR
    进场条件：月末月初效应为正且MACD柱状图>0
    出场条件：ATR追踪止损 / MACD柱状图持续为负
    优点：利用月末资金流入效应
    缺点：月末效应不总是出现
    """
    name = "mt_v282"
    warmup = 200
    freq = "1h"

    turn_window: int = 3
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._turn = None
        self._macd_hist = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
        self.neg_count = 0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        datetimes = context.get_full_datetime_array()
        self._turn = month_turn(closes, datetimes, window=self.turn_window)
        _, _, self._macd_hist = macd(closes, fast=12, slow=26, signal=9)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        turn_val = self._turn[i]
        hist_val = self._macd_hist[i]
        if np.isnan(atr_val) or np.isnan(turn_val) or np.isnan(hist_val) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return
            if hist_val < 0:
                self.neg_count += 1
            else:
                self.neg_count = 0

        if side == 0 and turn_val > 0 and hist_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
                self.neg_count = 0
        elif side == 1 and self.neg_count >= 5:
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
        return max(1, min(risk_lots, int(context.equity * 0.30 / margin)))

    def _reset(self):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
        self.neg_count = 0
