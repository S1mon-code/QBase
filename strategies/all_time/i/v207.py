import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.momentum.kama import kama
from indicators.regime.trend_persistence import trend_persistence


class AllTimeIV207(TimeSeriesStrategy):
    """
    策略简介：KAMA方向 + Trend Persistence过滤的日线策略。
    使用指标：KAMA(10)自适应均线方向，Trend Persistence过滤。
    进场条件：KAMA上升且TP>0.5做多；KAMA下降且TP>0.5做空。
    出场条件：ATR追踪止损 + KAMA反转。
    优点：KAMA自适应避免震荡市频繁交易。
    缺点：自适应特性可能在快速反转时反应不够。
    """
    name = "i_alltime_v207"
    warmup = 60
    freq = "daily"

    tp_threshold: float = 0.5
    atr_stop_mult: float = 4.5

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

        self._atr = atr(highs, lows, closes, period=14)
        self._kama = kama(closes, period=10)
        self._tp = trend_persistence(closes, max_lag=20, period=60)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._kama[i]) or np.isnan(self._tp[i]) or np.isnan(atr_val):
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

        if side == 0 and self._kama[i] > self._kama[i-1] and self._tp[i] > self.tp_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._kama[i] < self._kama[i-1] and self._tp[i] > self.tp_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._kama[i] < self._kama[i-1]:
            context.close_long()
            self._reset()

        elif side == -1 and self._kama[i] > self._kama[i-1]:
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
