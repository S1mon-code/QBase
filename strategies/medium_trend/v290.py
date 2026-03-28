import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.seasonality.holiday_proximity import holiday_effect
from indicators.momentum.coppock import coppock
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV290(TimeSeriesStrategy):
    """
    策略简介：假日效应+Coppock曲线长期底部检测。
    使用指标：Holiday Effect(252) + Coppock Curve(10,14,11) + ATR
    进场条件：假日效应非负且Coppock从负转正
    出场条件：ATR追踪止损 / Coppock拐头向下
    优点：假日前买入效应+长期底部信号
    缺点：Coppock信号极稀少
    """
    name = "mt_v290"
    warmup = 300
    freq = "1h"

    coppock_wma: int = 10
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._holiday = None
        self._coppock = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        datetimes = context.get_full_datetime_array()
        self._holiday = holiday_effect(closes, datetimes, lookback=252)
        self._coppock = coppock(closes, wma=self.coppock_wma, roc_long=14, roc_short=11)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        hol_val = self._holiday[i]
        cop_val = self._coppock[i]
        if np.isnan(atr_val) or np.isnan(hol_val) or np.isnan(cop_val) or atr_val <= 0:
            return
        if i < 1:
            return
        cop_prev = self._coppock[i - 1]
        if np.isnan(cop_prev):
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and hol_val >= 0 and cop_prev < 0 and cop_val >= 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and cop_val < cop_prev and cop_prev > 0:
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
