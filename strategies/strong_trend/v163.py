"""
QBase Strong Trend Strategy v163 — Online SGD Signal + Keltner Channel
=======================================================================

策略简介：在线随机梯度下降模型实时学习价格模式，当SGD预测信号为正
         且价格突破Keltner上轨时入场做多。

使用指标：
  - Online SGD Signal (lr=0.01, period=20): 在线学习趋势预测
  - Keltner Channel (ema=20, atr=10, mult=1.5): 通道突破
  - ATR (period=14): trailing stop

进场条件：
  1. SGD signal > 0（模型预测上涨）
  2. 收盘价 > Keltner上轨（突破上轨确认）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. 收盘价 < Keltner中轨（回落到通道内）

优点：在线学习实时适应市场变化
缺点：学习率影响适应速度和稳定性的平衡
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx
from indicators.ml.online_regression import online_sgd_signal
from indicators.trend.keltner import keltner
from indicators.volatility.atr import atr


class StrongTrendV163(TimeSeriesStrategy):
    """在线SGD趋势预测 + Keltner通道突破。"""
    name = "strong_trend_v163"
    warmup = 60
    freq = "daily"

    sgd_lr: float = 0.01
    sgd_period: int = 20
    kc_mult: float = 1.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._sgd = None
        self._kc_upper = None
        self._kc_mid = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        rsi_arr = rsi(closes, 14)
        adx_arr = adx(highs, lows, closes, 14)
        features = np.column_stack([rsi_arr, adx_arr])

        self._sgd = online_sgd_signal(closes, features, learning_rate=self.sgd_lr, period=self.sgd_period)
        self._kc_upper, self._kc_mid, _ = keltner(highs, lows, closes, ema=20, atr=10, mult=self.kc_mult)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        sgd_val = self._sgd[i]
        kc_up = self._kc_upper[i]
        kc_m = self._kc_mid[i]
        atr_val = self._atr[i]
        if np.isnan(sgd_val) or np.isnan(kc_up) or np.isnan(kc_m) or np.isnan(atr_val):
            return

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
        if side == 0 and sgd_val > 0 and price > kc_up:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and price < kc_m:
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
