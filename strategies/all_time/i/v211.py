import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.momentum.tsi import tsi
from indicators.regime.stationarity_score import stationarity


class AllTimeIV211(TimeSeriesStrategy):
    """
    策略简介：TSI信号交叉 + Stationarity过滤的日线趋势策略。
    使用指标：TSI(25,13,7)线/信号交叉，Stationarity Score过滤。
    进场条件：TSI>Signal且Stationarity<0.5做多；TSI<Signal且Stationarity<0.5做空。
    出场条件：ATR追踪止损 + TSI/Signal交叉反转。
    优点：TSI双重平滑降噪效果好。
    缺点：平滑导致信号滞后。
    """
    name = "i_alltime_v211"
    warmup = 60
    freq = "daily"

    stat_threshold: float = 0.5
    atr_stop_mult: float = 4.0

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
        self._tsi, self._tsi_signal = tsi(closes, long=25, short=13, signal=7)
        self._stat = stationarity(closes, period=60)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._tsi[i]) or np.isnan(self._tsi_signal[i]) or np.isnan(self._stat[i]) or np.isnan(atr_val):
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

        if side == 0 and self._tsi[i] > self._tsi_signal[i] and self._stat[i] < self.stat_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._tsi[i] < self._tsi_signal[i] and self._stat[i] < self.stat_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._tsi[i] < self._tsi_signal[i]:
            context.close_long()
            self._reset()

        elif side == -1 and self._tsi[i] > self._tsi_signal[i]:
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
