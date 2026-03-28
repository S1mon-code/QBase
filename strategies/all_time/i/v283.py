import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.regime.efficiency_ratio import efficiency_ratio
from indicators.momentum.stochastic import stochastic


class AllTimeIV283(TimeSeriesStrategy):
    """
    策略简介：Efficiency Ratio自适应 + Stochastic信号的30min策略。
    使用指标：ER(20)效率比率区分趋势/震荡，Stochastic(14,3)信号。
    进场条件：ER>0.4时%K>80做多（趋势追踪）；ER<0.2时%K<20做多（均值回归）。
    出场条件：ATR追踪止损 + Stochastic回归。
    优点：ER自适应切换两种模式。
    缺点：中间区域ER信号不明确。
    """
    name = "i_alltime_v283"
    warmup = 60
    freq = "30min"

    er_trend: float = 0.4
    er_range: float = 0.2
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
        self._er = efficiency_ratio(closes, period=20)
        self._stoch_k, self._stoch_d = stochastic(highs, lows, closes, k=14, d=3)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._er[i]) or np.isnan(self._stoch_k[i]) or np.isnan(atr_val):
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

        if side == 0 and (self._er[i] > self.er_trend and self._stoch_k[i] > 80) or (self._er[i] < self.er_range and self._stoch_k[i] < 20):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and (self._er[i] > self.er_trend and self._stoch_k[i] < 20) or (self._er[i] < self.er_range and self._stoch_k[i] > 80):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._stoch_k[i] < 50:
            context.close_long()
            self._reset()

        elif side == -1 and self._stoch_k[i] > 50:
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
