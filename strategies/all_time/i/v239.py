import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.momentum.momentum_accel import momentum_acceleration
from indicators.volume.volume_efficiency import volume_efficiency


class AllTimeIV239(TimeSeriesStrategy):
    """
    策略简介：Momentum Acceleration + Volume Efficiency的4h均值回归。
    使用指标：Momentum Acceleration(10,20)加速度反转，Volume Efficiency(20)。
    进场条件：加速度从负转正且VE<0做多；加速度从正转负且VE>0做空。
    出场条件：固定ATR止损 + 加速度回归0。
    优点：加速度反转是领先信号。
    缺点：加速度噪音大，需要配合过滤。
    """
    name = "i_alltime_v239"
    warmup = 60
    freq = "4h"

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
        self._accel = momentum_acceleration(closes, fast=10, slow=20)
        self._ve = volume_efficiency(closes, volumes, period=20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._accel[i]) or np.isnan(self._ve[i]) or np.isnan(atr_val):
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

        if side == 0 and self._accel[i] > 0 and self._accel[i-1] < 0 and self._ve[i] < 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price

        elif side == 0 and self._accel[i] < 0 and self._accel[i-1] > 0 and self._ve[i] > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price

        elif side == 1 and self._accel[i] < 0:
            context.close_long()
            self._reset()

        elif side == -1 and self._accel[i] > 0:
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
