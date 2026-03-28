"""
QBase Strong Trend Strategy v159 — Prophet-Like Piecewise Trend + Volume Spike
================================================================================

策略简介：类Prophet分段线性趋势检测变点和趋势方向，当趋势斜率为正
         且成交量放量时入场做多。

使用指标：
  - Prophet-Like Piecewise Trend (n_changepoints=5, period=252): 分段趋势
  - Volume Spike (period=20, threshold=2.0): 成交量放量检测
  - ATR (period=14): trailing stop

进场条件：
  1. Piecewise trend 上升（当前 > 前一bar）
  2. Volume Spike 触发（放量确认突破）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Piecewise trend 下降

优点：自动检测趋势变点，适应不同市场阶段
缺点：变点数量影响趋势灵活度
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.ml.prophet_like_trend import piecewise_trend
from indicators.volume.volume_spike import volume_spike
from indicators.volatility.atr import atr


class StrongTrendV159(TimeSeriesStrategy):
    """类Prophet分段趋势 + 成交量放量确认。"""
    name = "strong_trend_v159"
    warmup = 60
    freq = "daily"

    n_changepoints: int = 5
    trend_period: int = 252
    vol_spike_threshold: float = 2.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._trend = None
        self._vol_spike = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._trend = piecewise_trend(closes, n_changepoints=self.n_changepoints, period=self.trend_period)
        self._vol_spike = volume_spike(volumes, period=20, threshold=self.vol_spike_threshold)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        trend_val = self._trend[i]
        trend_prev = self._trend[i - 1] if i > 0 else np.nan
        spike_val = self._vol_spike[i]
        atr_val = self._atr[i]
        if np.isnan(trend_val) or np.isnan(trend_prev) or np.isnan(spike_val) or np.isnan(atr_val):
            return

        trend_rising = trend_val > trend_prev
        has_spike = spike_val > 0

        # === Stop Loss Check ===
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # === Entry ===
        if side == 0 and trend_rising and has_spike:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and not trend_rising:
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
        max_lots = int(context.equity * 0.30 / margin)
        return max(1, min(risk_lots, max_lots))

    def _reset(self):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
