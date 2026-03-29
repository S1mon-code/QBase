import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.momentum.connors_rsi import connors_rsi
from indicators.trend.keltner import keltner


class AllTimeIV227(TimeSeriesStrategy):
    """
    策略简介：Connors RSI + Keltner通道的4h均值回归策略。
    使用指标：Connors RSI(3,2,100)短周期超买超卖，Keltner(20,10,1.5)通道。
    进场条件：ConnorsRSI<10且价格<Keltner下轨做多；ConnorsRSI>90且价格>上轨做空。
    出场条件：固定ATR止损 + 价格回归Keltner中轨。
    优点：Connors RSI对短期极端行情非常敏感。
    缺点：短周期RSI噪音大。
    """
    name = "i_alltime_v227"
    warmup = 60
    freq = "4h"

    crsi_low: float = 10.0
    crsi_high: float = 90.0
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
        self._crsi = connors_rsi(closes, rsi=3, streak=2, pct_rank=100)
        self._kc_upper, self._kc_mid, self._kc_lower = keltner(highs, lows, closes)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._crsi[i]) or np.isnan(self._kc_upper[i]) or np.isnan(self._kc_lower[i]) or np.isnan(atr_val):
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

        if side == 0 and self._crsi[i] < self.crsi_low and price < self._kc_lower[i]:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price

        elif side == 0 and self._crsi[i] > self.crsi_high and price > self._kc_upper[i]:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price

        elif side == 1 and price > self._kc_mid[i]:
            context.close_long()
            self._reset()

        elif side == -1 and price < self._kc_mid[i]:
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
