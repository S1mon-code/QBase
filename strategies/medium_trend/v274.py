import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.structure.smart_money import smart_money_index
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV274(TimeSeriesStrategy):
    """
    策略简介：Smart Money Index跟踪策略，追踪机构资金动向。
    使用指标：Smart Money Index(20) + ATR
    进场条件：SMI持续上升且斜率为正
    出场条件：ATR追踪止损 / SMI拐头向下
    优点：跟踪机构行为，领先零售
    缺点：SMI信号可能有噪声
    """
    name = "mt_v274"
    warmup = 200
    freq = "1h"

    smi_period: int = 20
    slope_lookback: int = 5
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._smi = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        opens = context.get_full_open_array()
        volumes = context.get_full_volume_array()
        self._smi = smart_money_index(opens, closes, highs, lows, volumes, period=self.smi_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        smi_val = self._smi[i]
        if np.isnan(atr_val) or np.isnan(smi_val) or atr_val <= 0:
            return
        if i < self.slope_lookback:
            return
        smi_prev = self._smi[i - self.slope_lookback]
        smi_prev1 = self._smi[i - 1]
        if np.isnan(smi_prev) or np.isnan(smi_prev1):
            return
        smi_rising = smi_val > smi_prev
        smi_turning_down = smi_val < smi_prev1

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and smi_rising:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and smi_turning_down:
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
