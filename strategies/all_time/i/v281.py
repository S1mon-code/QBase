import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.trend.adx import adx as adx_indicator
from indicators.momentum.rsi import rsi as rsi_indicator


class AllTimeIV281(TimeSeriesStrategy):
    """
    策略简介：ADX区间自适应 + RSI方向的30min混合策略。
    使用指标：ADX(14)判断趋势/震荡，RSI(14)方向信号。
    进场条件：ADX>25时RSI>60做多（趋势）；ADX<20时RSI<30做多（回归）。
    出场条件：ATR追踪止损 + RSI回归中性。
    优点：自适应切换趋势跟踪和均值回归。
    缺点：ADX阈值附近切换频繁。
    """
    name = "i_alltime_v281"
    warmup = 60
    freq = "30min"

    adx_trend: float = 25.0
    adx_range: float = 20.0
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
        self._adx = adx_indicator(highs, lows, closes, period=14)
        self._rsi = rsi_indicator(closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._adx[i]) or np.isnan(self._rsi[i]) or np.isnan(atr_val):
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

        if side == 0 and (self._adx[i] > self.adx_trend and self._rsi[i] > 60) or (self._adx[i] < self.adx_range and self._rsi[i] < 30):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and (self._adx[i] > self.adx_trend and self._rsi[i] < 40) or (self._adx[i] < self.adx_range and self._rsi[i] > 70):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._rsi[i] < 50 and self._adx[i] > self.adx_trend:
            context.close_long()
            self._reset()

        elif side == -1 and self._rsi[i] > 50 and self._adx[i] > self.adx_trend:
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
