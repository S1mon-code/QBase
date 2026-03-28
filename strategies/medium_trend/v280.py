import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.structure.speculation_index import speculation_index
from indicators.structure.pvt_strength import pvt_strength
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV280(TimeSeriesStrategy):
    """
    策略简介：投机指数+PVT强度组合，低投机+强PVT=健康趋势。
    使用指标：Speculation Index(20) + PVT Strength(20) + ATR
    进场条件：投机指数适中且PVT强度>0
    出场条件：ATR追踪止损 / 投机指数极端或PVT转负
    优点：过滤过度投机的不健康行情
    缺点：投机指数阈值需要经验校准
    """
    name = "mt_v280"
    warmup = 200
    freq = "1h"

    spec_period: int = 20
    pvt_period: int = 20
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._spec = None
        self._pvt = None
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
        oi = context.get_full_oi_array()
        self._spec = speculation_index(volumes, oi, period=self.spec_period)
        self._pvt = pvt_strength(closes, volumes, period=self.pvt_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        spec_val = self._spec[i]
        pvt_val = self._pvt[i]
        if np.isnan(atr_val) or np.isnan(spec_val) or np.isnan(pvt_val) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Moderate speculation (not extreme) + positive PVT
        if side == 0 and 0.3 < spec_val < 2.0 and pvt_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and (spec_val > 3.0 or pvt_val < -0.5):
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
