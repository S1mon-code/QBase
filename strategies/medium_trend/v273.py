import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.structure.squeeze_detector import squeeze_probability
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV273(TimeSeriesStrategy):
    """
    策略简介：空头挤压概率检测策略，高挤压概率时做多。
    使用指标：Squeeze Probability(20) + ATR
    进场条件：挤压概率>0.6且价格创5日新高
    出场条件：ATR追踪止损 / 挤压概率回落至低位
    优点：捕捉空头被迫平仓的强势上涨
    缺点：挤压事件不频繁
    """
    name = "mt_v273"
    warmup = 200
    freq = "1h"

    squeeze_period: int = 20
    squeeze_threshold: float = 0.6
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._squeeze = None
        self._atr = None
        self._highs = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()
        self._squeeze = squeeze_probability(closes, oi, volumes, period=self.squeeze_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._highs = highs

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        sq_val = self._squeeze[i]
        if np.isnan(atr_val) or np.isnan(sq_val) or atr_val <= 0:
            return
        if i < 5:
            return
        recent_high = np.nanmax(self._highs[i - 5:i])
        at_high = price >= recent_high

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and sq_val > self.squeeze_threshold and at_high:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and sq_val < 0.2:
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
