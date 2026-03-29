import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.trend.zlema import zlema
from indicators.regime.momentum_regime import momentum_regime


class AllTimeIV215(TimeSeriesStrategy):
    """
    策略简介：ZLEMA交叉 + Momentum Regime过滤的日线策略。
    使用指标：ZLEMA(15/40)交叉信号，Momentum Regime过滤。
    进场条件：ZLEMA快>慢且Momentum Regime=1做多；快<慢且Regime=-1做空。
    出场条件：ATR追踪止损 + ZLEMA交叉反转。
    优点：ZLEMA零滞后特性提高入场时机。
    缺点：Momentum Regime可能在转折点滞后。
    """
    name = "i_alltime_v215"
    warmup = 60
    freq = "daily"

    fast_period: int = 15
    slow_period: int = 40
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
        self._zlema_fast = zlema(closes, self.fast_period)
        self._zlema_slow = zlema(closes, self.slow_period)
        self._mom_regime = momentum_regime(closes, fast=10, slow=60)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._zlema_fast[i]) or np.isnan(self._zlema_slow[i]) or np.isnan(self._mom_regime[i]) or np.isnan(atr_val):
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

        if side == 0 and self._zlema_fast[i] > self._zlema_slow[i] and self._mom_regime[i] == 1:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._zlema_fast[i] < self._zlema_slow[i] and self._mom_regime[i] == -1:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._zlema_fast[i] < self._zlema_slow[i]:
            context.close_long()
            self._reset()

        elif side == -1 and self._zlema_fast[i] > self._zlema_slow[i]:
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
