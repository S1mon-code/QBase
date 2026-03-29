import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.momentum.cmo import cmo
from indicators.momentum.dpo import dpo


class AllTimeIV226(TimeSeriesStrategy):
    """
    策略简介：CMO + DPO双振荡器的4h均值回归策略。
    使用指标：CMO(14)超买超卖，DPO(20)去趋势价格。
    进场条件：CMO<-50且DPO<0做多；CMO>50且DPO>0做空。
    出场条件：固定ATR止损 + CMO回归0。
    优点：DPO去除趋势成分，纯看周期性。
    缺点：去趋势后可能错过强趋势的回调。
    """
    name = "i_alltime_v226"
    warmup = 60
    freq = "4h"

    cmo_threshold: float = 50.0
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
        self._cmo = cmo(closes, period=14)
        self._dpo = dpo(closes, period=20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._cmo[i]) or np.isnan(self._dpo[i]) or np.isnan(atr_val):
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

        if side == 0 and self._cmo[i] < -self.cmo_threshold and self._dpo[i] < 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price

        elif side == 0 and self._cmo[i] > self.cmo_threshold and self._dpo[i] > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price

        elif side == 1 and self._cmo[i] > 0:
            context.close_long()
            self._reset()

        elif side == -1 and self._cmo[i] < 0:
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
