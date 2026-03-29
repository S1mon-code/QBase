"""
QBase Strong Trend Strategy v88 — Momentum Regime + Keltner Channel
=====================================================================

策略简介：Momentum Regime 识别市场动量状态（正/负/中性），
         Keltner Channel 突破确认趋势启动。

使用指标：
  - Momentum Regime (fast=10, slow=60): 动量 regime 状态
  - Keltner Channel (ema=20, atr=10, mult=1.5): 通道突破
  - ATR (period=14): trailing stop

进场条件：
  1. Momentum Regime > 0.5（正动量 regime）
  2. 收盘价 > Keltner Upper（突破上轨）

出场条件：
  1. ATR trailing stop（mult=4.0）
  2. 收盘价 < Keltner Middle（回落至中轨下方）

优点：Regime 过滤 + 通道突破双重确认，减少虚假突破
缺点：Keltner Channel 在高波动时通道过宽
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.momentum_regime import momentum_regime
from indicators.trend.keltner import keltner
from indicators.volatility.atr import atr


class StrongTrendV88(TimeSeriesStrategy):
    """Momentum Regime filter + Keltner Channel breakout."""
    name = "strong_trend_v88"
    warmup = 60
    freq = "daily"

    mom_fast: int = 10
    mom_slow: int = 60
    kc_ema: int = 20
    kc_mult: float = 1.5
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._mom_regime = None
        self._kc_upper = None
        self._kc_mid = None
        self._kc_lower = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._mom_regime = momentum_regime(closes, fast=self.mom_fast, slow=self.mom_slow)
        self._kc_upper, self._kc_mid, self._kc_lower = keltner(
            highs, lows, closes, ema=self.kc_ema, atr=10, mult=self.kc_mult
        )
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        mr_val = self._mom_regime[i]
        kc_u = self._kc_upper[i]
        kc_m = self._kc_mid[i]
        atr_val = self._atr[i]
        if np.isnan(mr_val) or np.isnan(kc_u) or np.isnan(kc_m) or np.isnan(atr_val):
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
        if side == 0 and mr_val > 0.5 and price > kc_u:
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
