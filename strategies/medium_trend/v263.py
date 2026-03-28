import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.volatility.ttm_squeeze import ttm_squeeze
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV263(TimeSeriesStrategy):
    """
    策略简介：Schaff趋势周期+TTM Squeeze组合，挤压释放+周期信号共振。
    使用指标：Schaff Trend Cycle(10,23,50) + TTM Squeeze + ATR
    进场条件：STC从低位上穿25且TTM Squeeze释放(squeeze=False)且momentum>0
    出场条件：ATR追踪止损 / STC下穿75
    优点：挤压后释放+趋势周期双确认
    缺点：TTM Squeeze在4h可能信号稀少
    """
    name = "mt_v263"
    warmup = 200
    freq = "4h"

    stc_period: int = 10
    stc_fast: int = 23
    atr_trail_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._stc = None
        self._squeeze = None
        self._squeeze_mom = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        self._stc = schaff_trend_cycle(closes, period=self.stc_period, fast=self.stc_fast, slow=50)
        self._squeeze, self._squeeze_mom = ttm_squeeze(highs, lows, closes)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        stc_val = self._stc[i]
        sq_val = self._squeeze[i]
        sq_mom = self._squeeze_mom[i]
        if np.isnan(atr_val) or np.isnan(stc_val) or np.isnan(sq_val) or np.isnan(sq_mom) or atr_val <= 0:
            return
        if i < 1:
            return
        stc_prev = self._stc[i - 1]
        if np.isnan(stc_prev):
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        squeeze_released = sq_val == 0
        if side == 0 and stc_prev < 25 and stc_val >= 25 and squeeze_released and sq_mom > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and stc_val < 75 and stc_prev >= 75:
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
