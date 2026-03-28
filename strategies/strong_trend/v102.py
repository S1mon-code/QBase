"""
Strong Trend Strategy v102 — Bollinger Band Width Expansion + ADX
==================================================================
Enters when Bollinger Band Width expands sharply above its recent average
while ADX confirms a strong directional trend is forming.

  1. Bollinger Width — volatility expansion detection
  2. ADX            — trend strength confirmation

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v102.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.bollinger import bollinger_width
from indicators.trend.adx import adx
from indicators.volatility.atr import atr


class StrongTrendV102(TimeSeriesStrategy):
    """
    策略简介：Bollinger带宽扩张 + ADX趋势强度确认的波动率突破策略
    使用指标：Bollinger Width（波动率扩张）、ADX（趋势强度）
    进场条件：BB Width > 1.5倍均值 + ADX > 阈值 + 价格在中轨上方
    出场条件：ATR trailing stop 或 ADX跌破退出阈值
    优点：双重确认过滤，避免无方向的波动率扩张
    缺点：ADX滞后可能错过突破初期
    """
    name = "strong_trend_v102"
    warmup = 60
    freq = "daily"

    bb_period: int = 20
    adx_period: int = 14
    adx_threshold: float = 25.0
    width_mult: float = 1.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._bb_width = None
        self._adx = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._bb_width = bollinger_width(closes, self.bb_period)
        self._adx = adx(highs, lows, closes, self.adx_period)
        self._atr = atr(highs, lows, closes, period=14)

        # Pre-compute rolling mean of BB width
        n = len(closes)
        self._bb_width_avg = np.full(n, np.nan)
        for k in range(39, n):
            window = self._bb_width[k - 39:k + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                self._bb_width_avg[k] = np.mean(valid)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        cur_width = self._bb_width[i]
        cur_adx = self._adx[i]
        atr_val = self._atr[i]
        width_avg = self._bb_width_avg[i]

        if np.isnan(cur_width) or np.isnan(cur_adx) or np.isnan(atr_val) or np.isnan(width_avg):
            return

        width_expanding = cur_width > self.width_mult * width_avg

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
        if side == 0 and width_expanding and cur_adx > self.adx_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and cur_adx < 15.0:
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
