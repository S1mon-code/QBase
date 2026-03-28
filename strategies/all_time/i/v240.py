import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.momentum.awesome_oscillator import ao
from indicators.volatility.normalized_range import normalized_range


class AllTimeIV240(TimeSeriesStrategy):
    """
    策略简介：Awesome Oscillator零轴交叉 + Normalized Range的4h均值回归。
    使用指标：AO(5,34)零轴交叉，Normalized Range(20)波动标准化。
    进场条件：AO从负上穿0且NR<均值做多；AO从正下穿0且NR<均值做空。
    出场条件：固定ATR止损 + AO反向穿越0轴。
    优点：AO零轴交叉信号清晰。
    缺点：零轴附近可能频繁震荡。
    """
    name = "i_alltime_v240"
    warmup = 60
    freq = "4h"

    nr_threshold: float = 0.02
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest = 0.0
        self.lowest = 999999.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()


        self._atr = atr(highs, lows, closes, period=14)
        self._ao = ao(highs, lows, fast=5, slow=34)
        self._nr = normalized_range(highs, lows, closes, period=20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._ao[i]) or np.isnan(self._nr[i]) or np.isnan(atr_val):
            return

        if side == 1:
            if price <= self.entry_price - self.atr_stop_mult * atr_val:
                context.close_long()
                self._reset()
                return

        elif side == -1:
            if price >= self.entry_price + self.atr_stop_mult * atr_val:
                context.close_short()
                self._reset()
                return

        if side == 0 and self._ao[i] > 0 and self._ao[i-1] < 0 and self._nr[i] < self.nr_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price

        elif side == 0 and self._ao[i] < 0 and self._ao[i-1] > 0 and self._nr[i] < self.nr_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price

        elif side == 1 and self._ao[i] < 0:
            context.close_long()
            self._reset()

        elif side == -1 and self._ao[i] > 0:
            context.close_short()
            self._reset()

    def _calc_lots(self, context, price, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_stop_mult * atr_val * spec.multiplier
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
        self.highest = 0.0
        self.lowest = 999999.0
