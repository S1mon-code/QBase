import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.momentum.chop_zone import choppiness_index
from indicators.momentum.cci import cci


class AllTimeIV285(TimeSeriesStrategy):
    """
    策略简介：Choppiness Index切换 + CCI方向的30min策略。
    使用指标：Choppiness(14)区分震荡/趋势，CCI(20)方向。
    进场条件：CI<38.2时CCI>100做多（趋势）；CI>61.8时CCI<-100做多（回归）。
    出场条件：ATR追踪止损 + CCI回归0。
    优点：CI分形阈值38.2/61.8有数学基础。
    缺点：CI转换区间信号模糊。
    """
    name = "i_alltime_v285"
    warmup = 60
    freq = "30min"

    ci_trend: float = 38.2
    ci_range: float = 61.8
    cci_threshold: float = 100.0
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
        self._ci = choppiness_index(highs, lows, closes, period=14)
        self._cci = cci(highs, lows, closes, period=20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._ci[i]) or np.isnan(self._cci[i]) or np.isnan(atr_val):
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

        if side == 0 and (self._ci[i] < self.ci_trend and self._cci[i] > self.cci_threshold) or (self._ci[i] > self.ci_range and self._cci[i] < -self.cci_threshold):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and (self._ci[i] < self.ci_trend and self._cci[i] < -self.cci_threshold) or (self._ci[i] > self.ci_range and self._cci[i] > self.cci_threshold):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and abs(self._cci[i]) < 50:
            context.close_long()
            self._reset()

        elif side == -1 and abs(self._cci[i]) < 50:
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
