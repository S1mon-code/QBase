import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.microstructure.trade_intensity import trade_intensity
from indicators.volume.close_location import close_location
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV295(TimeSeriesStrategy):
    """
    策略简介：交易强度+收盘位置组合，高交易强度+收在高位做多。
    使用指标：Trade Intensity(20) + Close Location + ATR
    进场条件：交易强度上升且收盘位置>0.7（收在日内高位）
    出场条件：ATR追踪止损 / 收盘位置跌破0.3
    优点：微观交易活跃度+价格位置确认
    缺点：收盘位置在趋势末期可能虚高
    """
    name = "mt_v295"
    warmup = 300
    freq = "30min"

    ti_period: int = 20
    cl_threshold: float = 0.7
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._ti = None
        self._cl = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        self._ti = trade_intensity(volumes, period=self.ti_period)
        self._cl = close_location(highs, lows, closes)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        ti_val = self._ti[i]
        cl_val = self._cl[i]
        if np.isnan(atr_val) or np.isnan(ti_val) or np.isnan(cl_val) or atr_val <= 0:
            return
        if i < 1:
            return
        ti_prev = self._ti[i - 1]
        if np.isnan(ti_prev):
            return
        ti_rising = ti_val > ti_prev

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and ti_rising and cl_val > self.cl_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and cl_val < 0.3:
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
