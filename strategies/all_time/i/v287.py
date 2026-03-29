import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.ml.kalman_trend import kalman_filter
from indicators.volatility.historical_vol import historical_volatility


class AllTimeIV287(TimeSeriesStrategy):
    """
    策略简介：Kalman Trend + Volatility自适应的30min策略。
    使用指标：Kalman Filter趋势方向，Historical Volatility(20)自适应止损。
    进场条件：Kalman上升且价格>Kalman做多；Kalman下降且价格<Kalman做空。
    出场条件：ATR追踪止损 + Kalman反转退出。
    优点：Kalman自适应平滑，波动率自适应止损。
    缺点：Kalman参数初始化影响大。
    """
    name = "i_alltime_v287"
    warmup = 60
    freq = "30min"

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
        self._kalman = kalman_filter(closes)
        self._hvol = historical_volatility(closes, period=20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._kalman[i]) or np.isnan(self._hvol[i]) or np.isnan(atr_val):
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

        if side == 0 and self._kalman[i] > self._kalman[i-1] and price > self._kalman[i]:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._kalman[i] < self._kalman[i-1] and price < self._kalman[i]:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and price < self._kalman[i]:
            context.close_long()
            self._reset()

        elif side == -1 and price > self._kalman[i]:
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
