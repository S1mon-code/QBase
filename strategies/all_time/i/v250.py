import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.trend.donchian import donchian
from indicators.volume.ad_line import ad_line


class AllTimeIV250(TimeSeriesStrategy):
    """
    策略简介：Donchian窄幅突破 + A/D Line方向的1h策略。
    使用指标：Donchian(15)窄通道快速突破，A/D Line方向确认。
    进场条件：价格突破DC上轨且AD上升做多；突破下轨且AD下降做空。
    出场条件：ATR追踪止损 + AD方向反转。
    优点：窄通道+AD方向确认提高入场频率。
    缺点：窄通道假突破较多。
    """
    name = "i_alltime_v250"
    warmup = 60
    freq = "1h"

    dc_period: int = 15
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
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(highs, lows, self.dc_period)
        self._ad = ad_line(highs, lows, closes, volumes)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._dc_upper[i]) or np.isnan(self._dc_lower[i]) or np.isnan(self._ad[i]) or np.isnan(atr_val):
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

        if side == 0 and price > self._dc_upper[i-1] and self._ad[i] > self._ad[i-1]:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and price < self._dc_lower[i-1] and self._ad[i] < self._ad[i-1]:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._ad[i] < self._ad[i-1]:
            context.close_long()
            self._reset()

        elif side == -1 and self._ad[i] > self._ad[i-1]:
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
