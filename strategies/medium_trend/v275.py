import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.structure.oi_breakout import oi_breakout
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV275(TimeSeriesStrategy):
    """
    策略简介：OI突破+Supertrend方向确认，OI异常放大确认趋势启动。
    使用指标：OI Breakout(20,2.0) + Supertrend(10,3.0) + ATR
    进场条件：OI突破发生且Supertrend方向为多
    出场条件：ATR追踪止损 / Supertrend翻空
    优点：OI突破=新资金入场，趋势更可靠
    缺点：OI突破事件稀少
    """
    name = "mt_v275"
    warmup = 200
    freq = "4h"

    oi_period: int = 20
    oi_threshold: float = 2.0
    atr_trail_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._oi_break = None
        self._st_dir = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        oi = context.get_full_oi_array()
        self._oi_break = oi_breakout(oi, period=self.oi_period, threshold=self.oi_threshold)
        _, self._st_dir = supertrend(highs, lows, closes, period=10, multiplier=3.0)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        oi_brk = self._oi_break[i]
        st_dir = self._st_dir[i]
        if np.isnan(atr_val) or np.isnan(oi_brk) or np.isnan(st_dir) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and oi_brk > 0 and st_dir == 1:
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
