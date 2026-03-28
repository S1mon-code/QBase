import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.microstructure.price_efficiency import price_efficiency_coefficient
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV293(TimeSeriesStrategy):
    """
    策略简介：价格效率系数+Supertrend方向确认，高效趋势做多。
    使用指标：Price Efficiency(20) + Supertrend(10,3.0) + ATR
    进场条件：价格效率>0.5（趋势而非随机游走）且Supertrend方向为多
    出场条件：ATR追踪止损 / Supertrend翻空
    优点：过滤随机波动的假趋势
    缺点：效率系数对周期长度敏感
    """
    name = "mt_v293"
    warmup = 300
    freq = "30min"

    pec_period: int = 20
    pec_threshold: float = 0.5
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._pec = None
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
        self._pec = price_efficiency_coefficient(closes, period=self.pec_period)
        _, self._st_dir = supertrend(highs, lows, closes, period=10, multiplier=3.0)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        pec_val = self._pec[i]
        st_dir = self._st_dir[i]
        if np.isnan(atr_val) or np.isnan(pec_val) or np.isnan(st_dir) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and pec_val > self.pec_threshold and st_dir == 1:
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
