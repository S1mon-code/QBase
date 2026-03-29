"""
QBase Strong Trend Strategy v185 — Hedging Pressure Proxy + Keltner
=====================================================================

策略简介：当对冲压力指标显示商业持仓者减仓（看涨信号）且价格突破Keltner通道上轨时做多。

使用指标：
  - Hedging Pressure (period=20): 对冲压力代理
  - Keltner Channel (ema=20, atr=10, mult=1.5): 波动通道
  - ATR (period=14): trailing stop

进场条件：
  1. Hedging Pressure < -0.5（对冲压力释放，看涨）
  2. 价格 > Keltner上轨（突破）

出场条件：
  1. ATR trailing stop (mult=4.0)
  2. 价格 < Keltner中轨（回落到均线以下）

优点：基本面结构 + 技术突破确认
缺点：对冲压力代理精度有限
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.spread.hedging_pressure import hedging_pressure
from indicators.trend.keltner import keltner
from indicators.volatility.atr import atr


class StrongTrendV185(TimeSeriesStrategy):
    """对冲压力 + Keltner通道突破策略。"""
    name = "strong_trend_v185"
    warmup = 60
    freq = "daily"

    hp_period: int = 20
    kc_ema: int = 20
    kc_mult: float = 1.5
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._hp = None
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
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()

        self._hp = hedging_pressure(closes, oi, volumes, period=self.hp_period)
        self._kc_upper, self._kc_mid, _ = keltner(highs, lows, closes,
                                                    ema=self.kc_ema, atr=10, mult=self.kc_mult)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        hp_val = self._hp[i]
        kc_u = self._kc_upper[i]
        kc_m = self._kc_mid[i]
        atr_val = self._atr[i]
        if np.isnan(hp_val) or np.isnan(kc_u) or np.isnan(kc_m) or np.isnan(atr_val):
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
        if side == 0 and hp_val < -0.5 and price > kc_u:
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
