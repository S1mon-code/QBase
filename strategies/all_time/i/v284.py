import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.volatility.hurst import hurst_exponent
from indicators.trend.ema import ema


class AllTimeIV284(TimeSeriesStrategy):
    """
    策略简介：Hurst Exponent自适应 + EMA交叉的30min混合策略。
    使用指标：Hurst(20)区分趋势(>0.5)/均值回归(<0.5)，EMA(10/30)交叉。
    进场条件：Hurst>0.55时EMA快>慢做多（趋势）；Hurst<0.45时EMA快<慢做多（回归）。
    出场条件：ATR追踪止损 + EMA交叉反转。
    优点：Hurst理论基础扎实。
    缺点：Hurst估计不稳定。
    """
    name = "i_alltime_v284"
    warmup = 60
    freq = "30min"

    hurst_trend: float = 0.55
    hurst_mr: float = 0.45
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
        self._hurst = hurst_exponent(closes, max_lag=20)
        self._ema_fast = ema(closes, 10)
        self._ema_slow = ema(closes, 30)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._hurst[i]) or np.isnan(self._ema_fast[i]) or np.isnan(self._ema_slow[i]) or np.isnan(atr_val):
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

        if side == 0 and (self._hurst[i] > self.hurst_trend and self._ema_fast[i] > self._ema_slow[i]) or (self._hurst[i] < self.hurst_mr and self._ema_fast[i] < self._ema_slow[i]):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and (self._hurst[i] > self.hurst_trend and self._ema_fast[i] < self._ema_slow[i]) or (self._hurst[i] < self.hurst_mr and self._ema_fast[i] > self._ema_slow[i]):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._ema_fast[i] < self._ema_slow[i] and self._hurst[i] > 0.5:
            context.close_long()
            self._reset()

        elif side == -1 and self._ema_fast[i] > self._ema_slow[i] and self._hurst[i] > 0.5:
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
