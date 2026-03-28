import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.momentum.cyber_cycle import cyber_cycle
from indicators.volatility.hurst import hurst_exponent


class AllTimeIV234(TimeSeriesStrategy):
    """
    策略简介：Cyber Cycle + Hurst Exponent的4h均值回归策略。
    使用指标：Cyber Cycle(0.07)周期信号，Hurst检测均值回归。
    进场条件：Cyber<-0.5且Hurst<0.45做多；Cyber>0.5且Hurst<0.45做空。
    出场条件：固定ATR止损 + Cyber Cycle过零退出。
    优点：Hurst<0.5确认均值回归特性。
    缺点：Cyber Cycle对alpha参数敏感。
    """
    name = "i_alltime_v234"
    warmup = 60
    freq = "4h"

    cyber_threshold: float = 0.5
    hurst_threshold: float = 0.45
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
        self._cyber = cyber_cycle(closes, alpha=0.07)
        self._hurst = hurst_exponent(closes, max_lag=20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._cyber[i]) or np.isnan(self._hurst[i]) or np.isnan(atr_val):
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

        if side == 0 and self._cyber[i] < -self.cyber_threshold and self._hurst[i] < self.hurst_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price

        elif side == 0 and self._cyber[i] > self.cyber_threshold and self._hurst[i] < self.hurst_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price

        elif side == 1 and self._cyber[i] > 0:
            context.close_long()
            self._reset()

        elif side == -1 and self._cyber[i] < 0:
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
