import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.trend.ema import ema
from indicators.trend.adx import adx as adx_indicator


class AllTimeIV201(TimeSeriesStrategy):
    """
    策略简介：EMA交叉 + ADX趋势过滤的日线趋势跟踪策略。
    使用指标：EMA(20/60)交叉判断方向，ADX(14)过滤震荡市。
    进场条件：EMA快线上穿慢线且ADX>25做多；下穿且ADX>25做空。
    出场条件：ATR追踪止损 + EMA反向交叉退出。
    优点：ADX过滤有效降低震荡市损耗。
    缺点：趋势启动初期可能滞后。
    """
    name = "i_alltime_v201"
    warmup = 60
    freq = "daily"

    fast_period: int = 20
    slow_period: int = 60
    adx_threshold: float = 25.0
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
        self._ema_fast = ema(closes, self.fast_period)
        self._ema_slow = ema(closes, self.slow_period)
        self._adx = adx_indicator(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._ema_fast[i]) or np.isnan(self._ema_slow[i]) or np.isnan(self._adx[i]) or np.isnan(atr_val):
            return

        # === Stop Loss (Long — trailing) ===
        if side == 1:
            self.highest = max(self.highest, price)
            if price <= self.highest - self.atr_stop_mult * atr_val:
                context.close_long()
                self._reset()
                return

        # === Stop Loss (Short — trailing) ===
        elif side == -1:
            self.lowest = min(self.lowest, price)
            if price >= self.lowest + self.atr_stop_mult * atr_val:
                context.close_short()
                self._reset()
                return

        # === Entry Long ===
        if side == 0 and self._ema_fast[i] > self._ema_slow[i] and self._adx[i] > self.adx_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        # === Entry Short ===
        elif side == 0 and self._ema_fast[i] < self._ema_slow[i] and self._adx[i] > self.adx_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        # === Signal Exit Long ===
        elif side == 1 and self._ema_fast[i] < self._ema_slow[i]:
            context.close_long()
            self._reset()

        # === Signal Exit Short ===
        elif side == -1 and self._ema_fast[i] > self._ema_slow[i]:
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
