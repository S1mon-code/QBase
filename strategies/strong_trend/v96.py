"""
QBase Strong Trend Strategy v96 — Correlation Breakdown + EMA Cross
=====================================================================

策略简介：Correlation Breakdown 检测相关性崩坏（市场 regime 变化信号），
         EMA Cross 确认新趋势方向后入场。

使用指标：
  - Correlation Breakdown (period=60): 相关性崩坏检测
  - EMA Cross (fast=10, slow=30): 均线交叉
  - ATR (period=14): trailing stop

进场条件：
  1. Correlation Breakdown 未触发（correlation_breakdown < threshold, 市场稳定趋势中）
  2. Fast EMA > Slow EMA（金叉）
  3. 收盘价 > Slow EMA

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Fast EMA < Slow EMA（死叉）

优点：Correlation Breakdown 能识别 regime 突变，避开混乱期
缺点：需要参考序列，单品种使用时参考自身收益率
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.correlation_breakdown import correlation_breakdown
from indicators.trend.ema import ema
from indicators.volatility.atr import atr


class StrongTrendV96(TimeSeriesStrategy):
    """Correlation Breakdown regime filter + EMA Cross."""
    name = "strong_trend_v96"
    warmup = 60
    freq = "daily"

    ema_fast: int = 10
    ema_slow: int = 30
    breakdown_thresh: float = 2.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._breakdown = None
        self._ema_fast = None
        self._ema_slow = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        # Use returns as both series for single-asset correlation breakdown
        n = len(closes)
        returns = np.full(n, np.nan)
        returns[1:] = np.diff(closes) / closes[:-1]
        # Use lagged returns as reference
        returns_lag = np.full(n, np.nan)
        returns_lag[2:] = returns[1:-1]

        self._breakdown = correlation_breakdown(returns, returns_lag, period=60, stress_threshold=self.breakdown_thresh)
        self._ema_fast = ema(closes, period=self.ema_fast)
        self._ema_slow = ema(closes, period=self.ema_slow)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        bd_val = self._breakdown[i]
        ef = self._ema_fast[i]
        es = self._ema_slow[i]
        atr_val = self._atr[i]
        if np.isnan(bd_val) or np.isnan(ef) or np.isnan(es) or np.isnan(atr_val):
            return

        # No breakdown = stable regime = OK to trade
        stable_regime = bd_val < 0.5

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
        if side == 0 and stable_regime and ef > es and price > es:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and ef < es:
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
