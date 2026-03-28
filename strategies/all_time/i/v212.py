import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.regime.regime_score import composite_regime


class AllTimeIV212(TimeSeriesStrategy):
    """
    策略简介：Schaff Trend Cycle + Composite Regime过滤的日线策略。
    使用指标：STC(10,23,50)趋势信号，Composite Regime Score过滤。
    进场条件：STC>75且Regime>0.3做多；STC<25且Regime>0.3做空。
    出场条件：ATR追踪止损 + STC反向穿越50退出。
    优点：STC反应比MACD快，Regime确保趋势环境。
    缺点：STC可能在震荡市频繁穿越阈值。
    """
    name = "i_alltime_v212"
    warmup = 60
    freq = "daily"

    stc_long: float = 75.0
    stc_short: float = 25.0
    regime_threshold: float = 0.3
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
        self._stc = schaff_trend_cycle(closes, period=10, fast=23, slow=50)
        self._regime = composite_regime(closes, highs, lows, period=20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._stc[i]) or np.isnan(self._regime[i]) or np.isnan(atr_val):
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

        if side == 0 and self._stc[i] > self.stc_long and self._regime[i] > self.regime_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._stc[i] < self.stc_short and self._regime[i] > self.regime_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._stc[i] < 50:
            context.close_long()
            self._reset()

        elif side == -1 and self._stc[i] > 50:
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
