"""
Strong Trend Strategy v124 — Historical Skewness Shift + Linear Regression
============================================================================
Detects shifts in return distribution skewness (from negative to positive
= bullish regime) combined with linear regression slope for trend direction.

  1. Historical Skewness  — positive skew shift = bullish distribution change
  2. Linear Regression    — slope and R-squared for trend confirmation

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v124.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.historical_skew import rolling_skewness
from indicators.trend.linear_regression import linear_regression_slope, r_squared
from indicators.volatility.atr import atr


class StrongTrendV124(TimeSeriesStrategy):
    """
    策略简介：历史偏度正向转变 + 线性回归趋势确认的策略
    使用指标：Rolling Skewness（滚动偏度）、Linear Regression（线性回归斜率+R²）
    进场条件：偏度从负转正 + 回归斜率 > 0 + R² > 0.5
    出场条件：ATR trailing stop 或 回归斜率 < 0
    优点：偏度正向转变是分布层面的结构性变化信号
    缺点：偏度计算需要大窗口（60+），滞后性强
    """
    name = "strong_trend_v124"
    warmup = 80
    freq = "daily"

    skew_period: int = 60
    lr_period: int = 20
    r2_threshold: float = 0.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._skew = None
        self._lr_slope = None
        self._r2 = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._skew = rolling_skewness(closes, period=self.skew_period)
        self._lr_slope = linear_regression_slope(closes, period=self.lr_period)
        self._r2 = r_squared(closes, period=self.lr_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        if i < 1:
            return

        cur_skew = self._skew[i]
        prev_skew = self._skew[i - 1]
        slope = self._lr_slope[i]
        r2_val = self._r2[i]
        atr_val = self._atr[i]

        if np.isnan(cur_skew) or np.isnan(prev_skew) or np.isnan(slope) or np.isnan(r2_val) or np.isnan(atr_val):
            return

        # Skewness shift: was negative, now positive
        skew_shift = prev_skew < 0 and cur_skew > 0
        # Or skewness rising significantly
        skew_rising = cur_skew > prev_skew and cur_skew > 0.5

        trend_confirmed = slope > 0 and r2_val > self.r2_threshold

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
        if side == 0 and (skew_shift or skew_rising) and trend_confirmed:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and slope < 0:
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
