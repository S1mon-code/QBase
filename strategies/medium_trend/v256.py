import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.regime.hurst_rs import hurst_rs
from indicators.trend.ema import ema
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV256(TimeSeriesStrategy):
    """
    策略简介：Hurst R/S指数趋势检测，H>0.5时市场有趋势记忆，配合EMA方向做多。
    使用指标：Hurst R/S + EMA(40) + ATR
    进场条件：Hurst > 0.55 且价格在EMA上方
    出场条件：ATR追踪止损 / Hurst回落至均值回归区间
    优点：有统计学基础，直接检测市场趋势性
    缺点：Hurst估算不稳定，短期噪声大
    """
    name = "mt_v256"
    warmup = 150
    freq = "daily"

    hurst_threshold: float = 0.55
    ema_period: int = 40
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._hurst = None
        self._ema = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        self._hurst = hurst_rs(closes, min_period=10, max_period=100)
        self._ema = ema(closes, period=self.ema_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        h_val = self._hurst[i]
        ema_val = self._ema[i]
        if np.isnan(atr_val) or np.isnan(h_val) or np.isnan(ema_val) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and h_val > self.hurst_threshold and price > ema_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and h_val < 0.45:
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
