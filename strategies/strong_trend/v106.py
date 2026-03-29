"""
Strong Trend Strategy v106 — ATR Ratio (Short/Long) Spike + Aroon
===================================================================
Uses the ratio of short-term ATR to long-term ATR to detect volatility
spikes, confirmed by Aroon indicator for trend direction.

  1. ATR Ratio — short/long ATR spike = volatility expansion
  2. Aroon     — trend direction confirmation

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v106.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.chop import atr_ratio
from indicators.trend.aroon import aroon
from indicators.volatility.atr import atr


class StrongTrendV106(TimeSeriesStrategy):
    """
    策略简介：ATR比率突增 + Aroon趋势确认的波动率突破策略
    使用指标：ATR Ratio（短/长期ATR比率）、Aroon（趋势方向）
    进场条件：ATR Ratio > 阈值 + Aroon Up > 70 + Aroon Osc > 0
    出场条件：ATR trailing stop 或 Aroon翻空
    优点：ATR Ratio反应快，Aroon提供方向过滤
    缺点：短期波动率突增可能是噪音而非趋势
    """
    name = "strong_trend_v106"
    warmup = 60
    freq = "daily"

    atr_short: int = 5
    atr_long: int = 20
    aroon_period: int = 25
    ratio_threshold: float = 1.3
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._atr_ratio = None
        self._aroon_up = None
        self._aroon_down = None
        self._aroon_osc = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._atr_ratio = atr_ratio(highs, lows, closes, short_period=self.atr_short, long_period=self.atr_long)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(highs, lows, self.aroon_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        ratio_val = self._atr_ratio[i]
        aroon_up = self._aroon_up[i]
        aroon_osc = self._aroon_osc[i]
        atr_val = self._atr[i]

        if np.isnan(ratio_val) or np.isnan(aroon_up) or np.isnan(atr_val):
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
        if side == 0 and ratio_val > self.ratio_threshold and aroon_up > 70 and aroon_osc > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and aroon_osc < -50:
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
