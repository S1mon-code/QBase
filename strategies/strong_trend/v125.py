"""
Strong Trend Strategy v125 — Historical Kurtosis Spike + Fractal Levels
=========================================================================
Detects spikes in return kurtosis (fat tails = extreme moves starting)
combined with fractal level breakouts for entry timing.

  1. Historical Kurtosis — fat tail detection (high kurtosis = extreme moves)
  2. Fractal Levels      — Williams fractal support/resistance for breakout

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v125.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.historical_kurtosis import rolling_kurtosis
from indicators.trend.fractal import fractal_levels
from indicators.volatility.atr import atr


class StrongTrendV125(TimeSeriesStrategy):
    """
    策略简介：历史峰度突增 + 分形水平突破的极端行情捕捉策略
    使用指标：Rolling Kurtosis（滚动峰度）、Fractal Levels（分形高低点）
    进场条件：Kurtosis > 阈值 (fat tails active) + 价格突破分形高点
    出场条件：ATR trailing stop 或 价格跌破分形低点
    优点：高峰度是极端行情的统计学前兆，分形是自然的阻力突破
    缺点：高峰度可能意味着双向极端，分形有固有延迟
    """
    name = "strong_trend_v125"
    warmup = 80
    freq = "daily"

    kurt_period: int = 60
    kurt_threshold: float = 2.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._kurt = None
        self._frac_high = None
        self._frac_low = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._kurt = rolling_kurtosis(closes, period=self.kurt_period)
        self._frac_high, self._frac_low = fractal_levels(highs, lows)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        cur_kurt = self._kurt[i]
        frac_h = self._frac_high[i]
        frac_l = self._frac_low[i]
        atr_val = self._atr[i]

        if np.isnan(cur_kurt) or np.isnan(frac_h) or np.isnan(atr_val):
            return

        kurt_spiking = cur_kurt > self.kurt_threshold
        above_fractal_high = price > frac_h

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
        if side == 0 and kurt_spiking and above_fractal_high:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and not np.isnan(frac_l) and price < frac_l:
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
