import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.ml.cusum_filter import cusum_event_filter
from indicators.trend.laguerre_filter import laguerre


class AllTimeIV297(TimeSeriesStrategy):
    """
    策略简介：CUSUM Event + Laguerre Filter方向的30min策略。
    使用指标：CUSUM Filter事件检测，Laguerre(0.8)平滑方向。
    进场条件：CUSUM事件触发且Laguerre上升做多；CUSUM且Laguerre下降做空。
    出场条件：ATR追踪止损 + Laguerre反转退出。
    优点：CUSUM检测结构性变化，Laguerre平滑方向。
    缺点：CUSUM阈值需要精细调整。
    """
    name = "i_alltime_v297"
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
        self._cusum = cusum_event_filter(closes, threshold=1.0)
        self._lag = laguerre(closes, gamma=0.8)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._cusum[i]) or np.isnan(self._lag[i]) or np.isnan(atr_val):
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

        if side == 0 and self._cusum[i] > 0.5 and self._lag[i] > self._lag[i-1]:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._cusum[i] > 0.5 and self._lag[i] < self._lag[i-1]:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._lag[i] < self._lag[i-1]:
            context.close_long()
            self._reset()

        elif side == -1 and self._lag[i] > self._lag[i-1]:
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
