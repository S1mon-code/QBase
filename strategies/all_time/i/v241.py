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
from indicators.volume.volume_spike import volume_spike


class AllTimeIV241(TimeSeriesStrategy):
    """
    策略简介：Donchian通道突破 + 成交量确认的4h突破策略。
    使用指标：Donchian(20)上下轨突破，Volume Spike(20,1.5)确认。
    进场条件：价格突破Donchian上轨且放量做多；突破下轨且放量做空。
    出场条件：ATR追踪止损 + 价格回到Donchian中轨。
    优点：Donchian通道突破是经典趋势入场。
    缺点：假突破在震荡市中频繁出现。
    """
    name = "i_alltime_v241"
    warmup = 60
    freq = "4h"

    dc_period: int = 20
    atr_stop_mult: float = 3.5

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
        self._vol_spike = volume_spike(volumes, 20, 1.5)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._dc_upper[i]) or np.isnan(self._dc_lower[i]) or np.isnan(self._vol_spike[i]) or np.isnan(atr_val):
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

        if side == 0 and price > self._dc_upper[i-1] and self._vol_spike[i] > 0.5:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and price < self._dc_lower[i-1] and self._vol_spike[i] > 0.5:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and price < self._dc_mid[i]:
            context.close_long()
            self._reset()

        elif side == -1 and price > self._dc_mid[i]:
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
