import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.seasonality.seasonal_momentum import seasonal_momentum
from indicators.momentum.rsi import rsi
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV281(TimeSeriesStrategy):
    """
    策略简介：季节性动量+RSI过滤，利用历史季节性模式做多。
    使用指标：Seasonal Momentum(3yr) + RSI(14) + ATR
    进场条件：季节性动量为正且RSI>50
    出场条件：ATR追踪止损 / 季节性动量转负
    优点：结合统计规律和当前动量
    缺点：季节性可能失效
    """
    name = "mt_v281"
    warmup = 200
    freq = "1h"

    rsi_period: int = 14
    rsi_threshold: int = 50
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._seasonal = None
        self._rsi = None
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
        self._seasonal = seasonal_momentum(closes, datetimes, lookback_years=3)
        self._rsi = rsi(closes, period=self.rsi_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        seas_val = self._seasonal[i]
        rsi_val = self._rsi[i]
        if np.isnan(atr_val) or np.isnan(seas_val) or np.isnan(rsi_val) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and seas_val > 0 and rsi_val > self.rsi_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and seas_val < -0.5:
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
