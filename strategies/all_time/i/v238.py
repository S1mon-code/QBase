import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.momentum.relative_vigor import relative_vigor_index
from indicators.microstructure.price_efficiency import price_efficiency_coefficient


class AllTimeIV238(TimeSeriesStrategy):
    """
    策略简介：Relative Vigor Index + 价格效率的4h均值回归策略。
    使用指标：RVI(10)振荡器，Price Efficiency Coefficient过滤。
    进场条件：RVI<-0.3且PEC<0.3做多；RVI>0.3且PEC<0.3做空。
    出场条件：固定ATR止损 + RVI回归0。
    优点：PEC低=价格效率低=均值回归环境。
    缺点：RVI计算依赖OHLC完整性。
    """
    name = "i_alltime_v238"
    warmup = 60
    freq = "4h"

    rvi_threshold: float = 0.3
    pec_threshold: float = 0.3
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest = 0.0
        self.lowest = 999999.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()


        self._atr = atr(highs, lows, closes, period=14)
        opens = context.get_full_open_array()
        self._rvi, _ = relative_vigor_index(opens, highs, lows, closes, period=10)
        self._pec = price_efficiency_coefficient(closes, period=20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._rvi[i]) or np.isnan(self._pec[i]) or np.isnan(atr_val):
            return

        if side == 1:
            if price <= self.entry_price - self.atr_stop_mult * atr_val:
                context.close_long()
                self._reset()
                return

        elif side == -1:
            if price >= self.entry_price + self.atr_stop_mult * atr_val:
                context.close_short()
                self._reset()
                return

        if side == 0 and self._rvi[i] < -self.rvi_threshold and self._pec[i] < self.pec_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price

        elif side == 0 and self._rvi[i] > self.rvi_threshold and self._pec[i] < self.pec_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price

        elif side == 1 and self._rvi[i] > 0:
            context.close_long()
            self._reset()

        elif side == -1 and self._rvi[i] < 0:
            context.close_short()
            self._reset()

    def _calc_lots(self, context, price, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_stop_mult * atr_val * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = price * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        max_lots = int(context.equity * 0.30 / margin)
        return max(1, min(risk_lots, max_lots))

    def _reset(self):
        self.entry_price = 0.0
        self.highest = 0.0
        self.lowest = 999999.0
