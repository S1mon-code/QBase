"""
QBase Strong Trend Strategy v151 — Kalman Trend Filter + Volume Momentum
=========================================================================

策略简介：利用 Kalman 滤波器提取价格趋势信号，当 Kalman 趋势斜率为正且
         成交量动量确认放量时入场做多。

使用指标：
  - Kalman Trend Filter: 自适应趋势提取，平滑噪声
  - Volume Momentum (period=14): 成交量动量确认
  - ATR (period=14): trailing stop

进场条件：
  1. Kalman 滤波值上升（当前 > 前一bar）
  2. 收盘价 > Kalman 滤波值（价格在趋势线上方）
  3. Volume Momentum > 1.0（放量确认）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. 收盘价 < Kalman 滤波值（跌破趋势线）

优点：Kalman 滤波自适应噪声，趋势提取平滑
缺点：过程噪声参数影响滤波灵敏度
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.ml.kalman_trend import kalman_filter
from indicators.volume.volume_momentum import volume_momentum
from indicators.volatility.atr import atr


class StrongTrendV151(TimeSeriesStrategy):
    """Kalman趋势滤波 + 成交量动量确认。"""
    name = "strong_trend_v151"
    warmup = 60
    freq = "daily"

    process_noise: float = 0.01
    vol_mom_period: int = 14
    vol_mom_threshold: float = 1.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._kalman = None
        self._vol_mom = None
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

        self._kalman = kalman_filter(closes, process_noise=self.process_noise)
        self._vol_mom = volume_momentum(volumes, period=self.vol_mom_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        kalman_val = self._kalman[i]
        kalman_prev = self._kalman[i - 1] if i > 0 else np.nan
        vm_val = self._vol_mom[i]
        atr_val = self._atr[i]
        if np.isnan(kalman_val) or np.isnan(kalman_prev) or np.isnan(vm_val) or np.isnan(atr_val):
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
        if side == 0 and kalman_val > kalman_prev and price > kalman_val and vm_val > self.vol_mom_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and price < kalman_val:
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
