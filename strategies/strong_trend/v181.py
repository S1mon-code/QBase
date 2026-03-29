"""
QBase Strong Trend Strategy v181 — Relative Strength (vs SMA benchmark) + Aroon
==================================================================================

策略简介：用收盘价与其SMA的比值作为相对强度信号，当价格持续跑赢自身均线且
         Aroon Up确认新高趋势时做多。

使用指标：
  - Relative Strength (asset vs SMA): 价格/SMA比值作为强度
  - Aroon (period=25): Aroon Up/Down趋势确认
  - ATR (period=14): trailing stop

进场条件：
  1. RS momentum > 0（价格加速跑赢均线）
  2. Aroon Up > 70（接近新高）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. Aroon Down > 70（接近新低）

优点：相对强度 + Aroon双重趋势确认
缺点：均线基准本身有滞后性
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.spread.relative_strength import relative_strength
from indicators.trend.aroon import aroon
from indicators.trend.sma import sma
from indicators.volatility.atr import atr


class StrongTrendV181(TimeSeriesStrategy):
    """相对强度(vs SMA) + Aroon趋势策略。"""
    name = "strong_trend_v181"
    warmup = 60
    freq = "daily"

    rs_period: int = 20
    sma_period: int = 50
    aroon_period: int = 25
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._rs_mom = None
        self._aroon_up = None
        self._aroon_down = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        benchmark = sma(closes, period=self.sma_period)
        _, self._rs_mom, _ = relative_strength(closes, benchmark, period=self.rs_period)
        self._aroon_up, self._aroon_down, _ = aroon(highs, lows, period=self.aroon_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        rs_m = self._rs_mom[i]
        ar_up = self._aroon_up[i]
        ar_dn = self._aroon_down[i]
        atr_val = self._atr[i]
        if np.isnan(rs_m) or np.isnan(ar_up) or np.isnan(ar_dn) or np.isnan(atr_val):
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
        if side == 0 and rs_m > 0.0 and ar_up > 70.0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and ar_dn > 70.0:
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
