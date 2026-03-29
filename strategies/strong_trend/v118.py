"""
Strong Trend Strategy v118 — GK Volatility Ratio (Fast/Slow) + Keltner
=========================================================================
Uses Garman-Klass volatility ratio (fast vs slow) to detect volatility
expansion regime, confirmed by price breaking above Keltner upper channel.

  1. GK Vol Ratio — fast/slow Garman-Klass vol ratio (expanding/contracting)
  2. Keltner      — channel breakout confirmation

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v118.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.gk_volatility_ratio import gk_vol_ratio
from indicators.trend.keltner import keltner
from indicators.volatility.atr import atr


class StrongTrendV118(TimeSeriesStrategy):
    """
    策略简介：GK波动率比率扩张 + Keltner通道突破的策略
    使用指标：GK Vol Ratio（快/慢GK波动率比）、Keltner Channel（通道突破）
    进场条件：GK Ratio > 1 (expanding regime) + 价格突破Keltner上轨
    出场条件：ATR trailing stop 或 价格跌破Keltner中轨
    优点：GK比率用OHLC数据效率高，Keltner比Bollinger更平滑
    缺点：双重技术指标可能过度过滤
    """
    name = "strong_trend_v118"
    warmup = 60
    freq = "daily"

    gk_fast: int = 10
    gk_slow: int = 60
    kc_ema: int = 20
    kc_mult: float = 1.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._gk_ratio = None
        self._gk_regime = None
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
        opens = context.get_full_open_array()

        self._gk_ratio, self._gk_regime = gk_vol_ratio(
            opens, highs, lows, closes, fast=self.gk_fast, slow=self.gk_slow
        )
        self._kc_upper, self._kc_mid, self._kc_lower = keltner(
            highs, lows, closes, ema=self.kc_ema, mult=self.kc_mult
        )
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        gk_ratio = self._gk_ratio[i]
        gk_regime = self._gk_regime[i]
        kc_upper = self._kc_upper[i]
        kc_mid = self._kc_mid[i]
        atr_val = self._atr[i]

        if np.isnan(gk_ratio) or np.isnan(kc_upper) or np.isnan(atr_val):
            return

        vol_expanding = gk_regime == 1  # expanding regime
        above_keltner = price >= kc_upper

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
        if side == 0 and vol_expanding and above_keltner:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and not np.isnan(kc_mid) and price < kc_mid:
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
