import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.trend.supertrend import supertrend
from indicators.volume.vroc import vroc


class AllTimeIV254(TimeSeriesStrategy):
    """
    策略简介：Supertrend突破 + Volume VROC确认的1h策略。
    使用指标：Supertrend(7,2.0)快速方向，VROC(14)成交量变化率。
    进场条件：Supertrend=1且VROC>50做多；Supertrend=-1且VROC>50做空。
    出场条件：ATR追踪止损 + Supertrend方向反转。
    优点：快速Supertrend+放量确认提供早期入场。
    缺点：短周期Supertrend假信号多。
    """
    name = "i_alltime_v254"
    warmup = 60
    freq = "1h"

    st_period: int = 7
    st_mult: float = 2.0
    vroc_threshold: float = 50.0
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
        _, self._st_dir = supertrend(highs, lows, closes, self.st_period, self.st_mult)
        self._vroc = vroc(volumes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._st_dir[i]) or np.isnan(self._vroc[i]) or np.isnan(atr_val):
            return

        if side == 1:
            self.highest = max(self.highest, price)
            if price <= self.highest - self.atr_stop_mult * atr_val:
                context.close_long()
                self._reset()
                return

        elif side == -1:
            self.lowest = min(self.lowest, price)
            if price >= self.lowest + self.atr_stop_mult * atr_val:
                context.close_short()
                self._reset()
                return

        if side == 0 and self._st_dir[i] == 1 and self._vroc[i] > self.vroc_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._st_dir[i] == -1 and self._vroc[i] > self.vroc_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._st_dir[i] == -1:
            context.close_long()
            self._reset()

        elif side == -1 and self._st_dir[i] == 1:
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
