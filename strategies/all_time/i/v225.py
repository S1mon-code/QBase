import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.momentum.stoch_rsi import stoch_rsi
from indicators.volatility.std_dev import z_score


class AllTimeIV225(TimeSeriesStrategy):
    """
    策略简介：Stochastic RSI + Z-Score的4h均值回归策略。
    使用指标：Stochastic RSI(14,14,3,3)超买超卖，Z-Score(20)偏离度。
    进场条件：StochRSI %K<0.1且Z<-2做多；%K>0.9且Z>2做空。
    出场条件：固定ATR止损 + Z-Score回归0。
    优点：StochRSI对极端行情更敏感。
    缺点：Z-Score假设正态分布可能不成立。
    """
    name = "i_alltime_v225"
    warmup = 60
    freq = "4h"

    srsi_low: float = 0.1
    srsi_high: float = 0.9
    z_threshold: float = 2.0
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
        self._srsi_k, self._srsi_d = stoch_rsi(closes, rsi=14, stoch=14, k=3, d=3)
        self._zscore = z_score(closes, period=20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._srsi_k[i]) or np.isnan(self._zscore[i]) or np.isnan(atr_val):
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

        if side == 0 and self._srsi_k[i] < self.srsi_low and self._zscore[i] < -self.z_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price

        elif side == 0 and self._srsi_k[i] > self.srsi_high and self._zscore[i] > self.z_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price

        elif side == 1 and self._zscore[i] > 0:
            context.close_long()
            self._reset()

        elif side == -1 and self._zscore[i] < 0:
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
