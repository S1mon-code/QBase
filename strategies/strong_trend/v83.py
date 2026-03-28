"""
QBase Strong Trend Strategy v83 — Fractal Dimension + PPO
==========================================================

策略简介：Fractal Dimension 低值（< 1.5）表示市场处于趋势状态，
         PPO 确认动量方向后入场做多。

使用指标：
  - Fractal Dimension (period=60): FD < 1.5 = 趋势，FD > 1.5 = 随机
  - PPO (fast=12, slow=26, signal=9): 动量方向确认
  - ATR (period=14): trailing stop

进场条件：
  1. Fractal Dimension < 1.45（趋势 regime）
  2. PPO histogram > 0（正动量）
  3. PPO line > signal line（动量加速）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Fractal Dimension > 1.6（随机 regime）

优点：Fractal Dimension 直接测量市场的分形特征，理论严谨
缺点：计算量较大，对短期噪音敏感
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.fractal_dimension import fractal_dim
from indicators.momentum.ppo import ppo
from indicators.volatility.atr import atr


class StrongTrendV83(TimeSeriesStrategy):
    """Low Fractal Dimension (trending) + PPO momentum."""
    name = "strong_trend_v83"
    warmup = 60
    freq = "daily"

    fd_threshold: float = 1.45
    ppo_fast: int = 12
    ppo_slow: int = 26
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._fd = None
        self._ppo_line = None
        self._ppo_signal = None
        self._ppo_hist = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._fd = fractal_dim(closes, period=60)
        self._ppo_line, self._ppo_signal, self._ppo_hist = ppo(closes, fast=self.ppo_fast, slow=self.ppo_slow, signal=9)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        fd_val = self._fd[i]
        ppo_h = self._ppo_hist[i]
        ppo_l = self._ppo_line[i]
        ppo_s = self._ppo_signal[i]
        atr_val = self._atr[i]
        if np.isnan(fd_val) or np.isnan(ppo_h) or np.isnan(ppo_l) or np.isnan(atr_val):
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
        if side == 0 and fd_val < self.fd_threshold and ppo_h > 0 and ppo_l > ppo_s:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and fd_val > 1.6:
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
