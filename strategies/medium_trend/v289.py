import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.seasonality.weekday_effect import weekday_effect
from indicators.momentum.stochastic import stochastic
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV289(TimeSeriesStrategy):
    """
    策略简介：星期效应+随机指标组合，利用特定交易日倾向做多。
    使用指标：Weekday Effect(252) + Stochastic(14,3) + ATR
    进场条件：星期效应为正且随机指标%K从超卖区上穿%D
    出场条件：ATR追踪止损 / %K下穿%D且在超买区
    优点：日历效应+超卖反弹动量
    缺点：星期效应不稳定
    """
    name = "mt_v289"
    warmup = 300
    freq = "1h"

    stoch_k: int = 14
    stoch_d: int = 3
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._weekday = None
        self._stoch_k = None
        self._stoch_d = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        datetimes = context.get_full_datetime_array()
        self._weekday = weekday_effect(closes, datetimes, lookback=252)
        self._stoch_k, self._stoch_d = stochastic(highs, lows, closes, k=self.stoch_k, d=self.stoch_d)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        wd_val = self._weekday[i]
        sk = self._stoch_k[i]
        sd = self._stoch_d[i]
        if np.isnan(atr_val) or np.isnan(wd_val) or np.isnan(sk) or np.isnan(sd) or atr_val <= 0:
            return
        if i < 1:
            return
        sk_prev = self._stoch_k[i - 1]
        sd_prev = self._stoch_d[i - 1]
        if np.isnan(sk_prev) or np.isnan(sd_prev):
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and wd_val > 0 and sk_prev < sd_prev and sk >= sd and sk < 30:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and sk > 80 and sk < sd:
            context.close_long()
            self._reset()

    def _calc_lots(self, context, price, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_trail_mult * atr_val * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = price * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        return max(1, min(risk_lots, int(context.equity * 0.30 / margin)))

    def _reset(self):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
