"""
QBase Strong Trend Strategy v82 — Sample Entropy + Volume Spike
================================================================

策略简介：Sample Entropy 低值表示价格序列规律性强（趋势特征），
         结合 Volume Spike 确认量能异常放大后入场。

使用指标：
  - Sample Entropy (m=2, r_mult=0.2, period=60): 低值 = 规律/趋势
  - Volume Spike (period=20, threshold=2.0): 成交量异常放大
  - ATR (period=14): trailing stop

进场条件：
  1. Sample Entropy < 1.0（低熵 = 趋势 regime）
  2. Volume Spike == 1（量能突破确认）
  3. 收盘价 > 前一日收盘价（方向过滤）

出场条件：
  1. ATR trailing stop（mult=4.0）
  2. Sample Entropy > 1.8（高熵 = 随机 regime）

优点：Sample Entropy 从信息论角度衡量趋势，与传统指标低相关
缺点：熵值对参数 m 和 r 敏感，需要调参
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.sample_entropy import sample_entropy
from indicators.volume.volume_spike import volume_spike
from indicators.volatility.atr import atr


class StrongTrendV82(TimeSeriesStrategy):
    """Low Sample Entropy (trending) + Volume Spike confirmation."""
    name = "strong_trend_v82"
    warmup = 60
    freq = "daily"

    entropy_threshold: float = 1.0
    vol_spike_period: int = 20
    vol_spike_thresh: float = 2.0
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._entropy = None
        self._vol_spike = None
        self._atr = None
        self._closes = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._entropy = sample_entropy(closes, m=2, r_mult=0.2, period=60)
        self._vol_spike = volume_spike(volumes, period=self.vol_spike_period, threshold=self.vol_spike_thresh)
        self._atr = atr(highs, lows, closes, period=14)
        self._closes = closes

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        ent_val = self._entropy[i]
        spike_val = self._vol_spike[i]
        atr_val = self._atr[i]
        if np.isnan(ent_val) or np.isnan(spike_val) or np.isnan(atr_val) or i < 1:
            return

        prev_close = self._closes[i - 1]

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
        if side == 0 and ent_val < self.entropy_threshold and spike_val == 1 and price > prev_close:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and ent_val > 1.8:
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
