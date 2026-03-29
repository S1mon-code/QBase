import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.trend.supertrend import supertrend
from indicators.volume.cmf import cmf
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV262(TimeSeriesStrategy):
    """
    策略简介：Supertrend+CMF资金流组合，趋势方向+资金流入双重确认。
    使用指标：Supertrend(10,3.0) + CMF(20) + ATR
    进场条件：Supertrend方向翻多且CMF>0
    出场条件：ATR追踪止损 / Supertrend翻空
    优点：经典趋势信号+资金流质量过滤
    缺点：CMF在低量时不稳定
    """
    name = "mt_v262"
    warmup = 200
    freq = "4h"

    st_period: int = 10
    st_mult: float = 3.0
    cmf_period: int = 20
    atr_trail_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._st_line = None
        self._st_dir = None
        self._cmf = None
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
        self._st_line, self._st_dir = supertrend(highs, lows, closes, period=self.st_period, multiplier=self.st_mult)
        self._cmf = cmf(highs, lows, closes, volumes, period=self.cmf_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        st_dir = self._st_dir[i]
        cmf_val = self._cmf[i]
        if np.isnan(atr_val) or np.isnan(st_dir) or np.isnan(cmf_val) or atr_val <= 0:
            return
        if i < 1:
            return
        prev_dir = self._st_dir[i - 1]
        if np.isnan(prev_dir):
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and prev_dir == -1 and st_dir == 1 and cmf_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and st_dir == -1:
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
