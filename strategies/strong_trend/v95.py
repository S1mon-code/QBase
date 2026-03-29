"""
QBase Strong Trend Strategy v95 — Vol Regime Markov + Donchian Channel
========================================================================

策略简介：基于简化 Markov 模型的 Vol Regime 检测低/高波动率状态，
         低波后转高波配合 Donchian Channel 突破入场。

使用指标：
  - Vol Regime Markov (period=60): regime 状态（低/高波动）
  - Donchian Channel (period=20): 通道突破
  - ATR (period=14): trailing stop

进场条件：
  1. Vol Regime 处于高波动状态（趋势启动 regime）
  2. 收盘价 > Donchian Upper（突破上轨）

出场条件：
  1. ATR trailing stop（mult=4.0）
  2. 收盘价 < Donchian Lower（跌破下轨）

优点：Markov regime 检测能识别波动率状态转换
缺点：简化的 Markov 模型可能延迟识别 regime 切换
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.vol_regime_markov import vol_regime_simple
from indicators.trend.donchian import donchian
from indicators.volatility.atr import atr


class StrongTrendV95(TimeSeriesStrategy):
    """Vol Regime Markov state + Donchian Channel breakout."""
    name = "strong_trend_v95"
    warmup = 60
    freq = "daily"

    dc_period: int = 20
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._vol_regime = None
        self._dc_upper = None
        self._dc_lower = None
        self._dc_mid = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._vol_regime = vol_regime_simple(closes, period=60)
        self._dc_upper, self._dc_lower, self._dc_mid = donchian(highs, lows, period=self.dc_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        vr_val = self._vol_regime[i]
        dc_u = self._dc_upper[i]
        dc_l = self._dc_lower[i]
        atr_val = self._atr[i]
        if np.isnan(vr_val) or np.isnan(dc_u) or np.isnan(dc_l) or np.isnan(atr_val):
            return

        # Vol regime: assume high vol state = value > 0.5
        is_high_vol = vr_val > 0.5

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
        if side == 0 and is_high_vol and price > dc_u:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and price < dc_l:
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
