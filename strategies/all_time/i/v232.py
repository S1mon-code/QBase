import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.momentum.reflex import reflex
from indicators.regime.mean_crossing_rate import mean_crossing


class AllTimeIV232(TimeSeriesStrategy):
    """
    策略简介：Reflex指标 + Mean Crossing Rate的4h均值回归策略。
    使用指标：Reflex(20)周期识别，Mean Crossing Rate确认震荡环境。
    进场条件：Reflex<-0.5且MCR>0.3做多；Reflex>0.5且MCR>0.3做空。
    出场条件：固定ATR止损 + Reflex回归0。
    优点：Reflex是领先指标，MCR确认震荡。
    缺点：Reflex在趋势市中失效。
    """
    name = "i_alltime_v232"
    warmup = 60
    freq = "4h"

    reflex_threshold: float = 0.5
    mcr_threshold: float = 0.3
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
        self._reflex = reflex(closes, period=20)
        self._mcr = mean_crossing(closes, period=60)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._reflex[i]) or np.isnan(self._mcr[i]) or np.isnan(atr_val):
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

        if side == 0 and self._reflex[i] < -self.reflex_threshold and self._mcr[i] > self.mcr_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price

        elif side == 0 and self._reflex[i] > self.reflex_threshold and self._mcr[i] > self.mcr_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price

        elif side == 1 and self._reflex[i] > 0:
            context.close_long()
            self._reset()

        elif side == -1 and self._reflex[i] < 0:
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
