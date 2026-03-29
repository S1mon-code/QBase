import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.ml.wavelet_decompose import wavelet_features
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV259(TimeSeriesStrategy):
    """
    策略简介：小波分解趋势策略，提取低频趋势成分过滤高频噪声。
    使用指标：Wavelet Features(db4, level=4) + ATR
    进场条件：小波趋势成分持续上升
    出场条件：ATR追踪止损 / 趋势成分拐头
    优点：多尺度分解，自动分离趋势和噪声
    缺点：边界效应，最新数据点估计不稳定
    """
    name = "mt_v259"
    warmup = 120
    freq = "daily"

    wavelet_level: int = 4
    slope_lookback: int = 5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._wavelet = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        self._wavelet = wavelet_features(closes, wavelet="db4", level=self.wavelet_level)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        wv_val = self._wavelet[i]
        if np.isnan(atr_val) or np.isnan(wv_val) or atr_val <= 0:
            return
        if i < self.slope_lookback:
            return
        wv_prev = self._wavelet[i - self.slope_lookback]
        if np.isnan(wv_prev):
            return
        wv_rising = wv_val > wv_prev

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and wv_rising and price > wv_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and not wv_rising:
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
