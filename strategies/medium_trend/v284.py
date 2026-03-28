import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.seasonality.intraweek_pattern import intraweek_momentum
from indicators.momentum.momentum_accel import momentum_acceleration
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV284(TimeSeriesStrategy):
    """
    策略简介：周内模式+动量加速度组合。
    使用指标：Intraweek Momentum(52) + Momentum Acceleration(10,20) + ATR
    进场条件：周内动量为正且动量加速度>0
    出场条件：ATR追踪止损 / 动量加速度持续为负
    优点：周内周期规律+加速确认
    缺点：周内模式在趋势市中可能失效
    """
    name = "mt_v284"
    warmup = 300
    freq = "1h"

    mom_fast: int = 10
    mom_slow: int = 20
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._intraweek = None
        self._mom_accel = None
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
        self._intraweek = intraweek_momentum(closes, datetimes, lookback=52)
        self._mom_accel = momentum_acceleration(closes, fast=self.mom_fast, slow=self.mom_slow)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        iw_val = self._intraweek[i]
        ma_val = self._mom_accel[i]
        if np.isnan(atr_val) or np.isnan(iw_val) or np.isnan(ma_val) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and iw_val > 0 and ma_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and ma_val < -0.5:
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
