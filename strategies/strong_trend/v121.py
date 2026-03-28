"""
Strong Trend Strategy v121 — Volatility Ratio Spike + Volume Momentum
=======================================================================
Uses Schwager's Volatility Ratio (TR/ATR) to detect breakout bars,
confirmed by volume momentum for institutional participation.

  1. Volatility Ratio — TR/ATR spike indicates potential breakout
  2. Volume Momentum  — confirms buying conviction behind the move

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v121.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.vol_ratio import volatility_ratio
from indicators.volume.volume_momentum import volume_momentum
from indicators.volatility.atr import atr


class StrongTrendV121(TimeSeriesStrategy):
    """
    策略简介：波动率比率突增 + 成交量动量确认的突破策略
    使用指标：Volatility Ratio（TR/ATR比率）、Volume Momentum（量能动量）
    进场条件：VR > 阈值 + Volume Momentum > 1 + 价格上涨
    出场条件：ATR trailing stop 或 Volume Momentum < 0.5
    优点：VR直接衡量当日波幅vs历史，量价共振过滤假突破
    缺点：VR为单日指标，可能触发过于频繁
    """
    name = "strong_trend_v121"
    warmup = 60
    freq = "daily"

    vr_period: int = 14
    vm_period: int = 14
    vr_threshold: float = 1.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._vr = None
        self._vm = None
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

        self._vr = volatility_ratio(highs, lows, closes, period=self.vr_period)
        self._vm = volume_momentum(volumes, self.vm_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._closes = closes

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        if i < 1:
            return

        vr_val = self._vr[i]
        vm_val = self._vm[i]
        atr_val = self._atr[i]

        if np.isnan(vr_val) or np.isnan(vm_val) or np.isnan(atr_val):
            return

        price_up = price > self._closes[i - 1]

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
        if side == 0 and vr_val > self.vr_threshold and vm_val > 1.0 and price_up:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and vm_val < 0.5:
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
