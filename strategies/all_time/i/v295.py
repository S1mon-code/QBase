import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.regime.sample_entropy import sample_entropy
from indicators.momentum.stochastic import kdj


class AllTimeIV295(TimeSeriesStrategy):
    """
    策略简介：Sample Entropy自适应 + KDJ方向的30min策略。
    使用指标：Sample Entropy(60)复杂度判断，KDJ(9)方向。
    进场条件：低熵时J>80做多（趋势）；高熵时J<0做多（回归）。
    出场条件：ATR追踪止损 + J回归50。
    优点：熵值从信息论角度判断市场状态。
    缺点：熵值估计对参数敏感。
    """
    name = "i_alltime_v295"
    warmup = 60
    freq = "30min"

    entropy_low: float = 0.5
    entropy_high: float = 1.5
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
        self._entropy = sample_entropy(closes, m=2, r_mult=0.2, period=60)
        _, _, self._j = kdj(highs, lows, closes, period=9, k=3, d=3)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._entropy[i]) or np.isnan(self._j[i]) or np.isnan(atr_val):
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

        if side == 0 and (self._entropy[i] < self.entropy_low and self._j[i] > 80) or (self._entropy[i] > self.entropy_high and self._j[i] < 0):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and (self._entropy[i] < self.entropy_low and self._j[i] < 20) or (self._entropy[i] > self.entropy_high and self._j[i] > 100):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._j[i] < 50:
            context.close_long()
            self._reset()

        elif side == -1 and self._j[i] > 50:
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
