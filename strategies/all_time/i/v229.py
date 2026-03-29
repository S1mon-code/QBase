import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.momentum.ultimate_oscillator import ultimate_oscillator
from indicators.volatility.ttm_squeeze import ttm_squeeze


class AllTimeIV229(TimeSeriesStrategy):
    """
    策略简介：Ultimate Oscillator + TTM Squeeze的4h均值回归策略。
    使用指标：Ultimate Oscillator(7,14,28)超买超卖，TTM Squeeze确认压缩。
    进场条件：UO<30且Squeeze=True做多；UO>70且Squeeze=True做空。
    出场条件：固定ATR止损 + UO回归50。
    优点：UO多时间框架降噪，Squeeze确认蓄力。
    缺点：Squeeze后可能趋势突破而非回归。
    """
    name = "i_alltime_v229"
    warmup = 60
    freq = "4h"

    uo_oversold: float = 30.0
    uo_overbought: float = 70.0
    atr_stop_mult: float = 3.0

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
        self._uo = ultimate_oscillator(highs, lows, closes, 7, 14, 28)
        self._squeeze, _ = ttm_squeeze(highs, lows, closes)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._uo[i]) or np.isnan(self._squeeze[i]) or np.isnan(atr_val):
            return

        if side == 1:
            if price <= self.entry_price - self.atr_stop_mult * atr_val:
                context.close_long()
                self._reset()
                return

        elif side == -1:
            if price >= self.entry_price + self.atr_stop_mult * atr_val:
                context.close_short()
                self._reset()
                return

        if side == 0 and self._uo[i] < self.uo_oversold and self._squeeze[i] > 0.5:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price

        elif side == 0 and self._uo[i] > self.uo_overbought and self._squeeze[i] > 0.5:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price

        elif side == 1 and self._uo[i] > 50:
            context.close_long()
            self._reset()

        elif side == -1 and self._uo[i] < 50:
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
