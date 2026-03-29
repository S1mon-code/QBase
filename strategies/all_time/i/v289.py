import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.regime.variance_ratio import variance_ratio_test
from indicators.momentum.williams_r import williams_r


class AllTimeIV289(TimeSeriesStrategy):
    """
    策略简介：Variance Ratio自适应 + Williams %R的30min策略。
    使用指标：VR(60,5)随机性检验，Williams %R(14)方向。
    进场条件：VR>1.1时WR>-20做多（趋势）；VR<0.9时WR<-80做多（回归）。
    出场条件：ATR追踪止损 + WR回归中性。
    优点：VR能统计区分趋势和均值回归。
    缺点：VR估计需要足够样本量。
    """
    name = "i_alltime_v289"
    warmup = 60
    freq = "30min"

    vr_trend: float = 1.1
    vr_mr: float = 0.9
    atr_stop_mult: float = 2.5

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
        self._vr = variance_ratio_test(closes, period=60, holding=5)
        self._wr = williams_r(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._vr[i]) or np.isnan(self._wr[i]) or np.isnan(atr_val):
            return

        if side == 1:
            self.highest = max(self.highest, price)
            if price <= self.highest - self.atr_stop_mult * atr_val:
                context.close_long()
                self._reset()
                return

        elif side == -1:
            self.lowest = min(self.lowest, price)
            if price >= self.lowest + self.atr_stop_mult * atr_val:
                context.close_short()
                self._reset()
                return

        if side == 0 and (self._vr[i] > self.vr_trend and self._wr[i] > -20) or (self._vr[i] < self.vr_mr and self._wr[i] < -80):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and (self._vr[i] > self.vr_trend and self._wr[i] < -80) or (self._vr[i] < self.vr_mr and self._wr[i] > -20):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._wr[i] < -50 and self._vr[i] > 1.0:
            context.close_long()
            self._reset()

        elif side == -1 and self._wr[i] > -50 and self._vr[i] > 1.0:
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
