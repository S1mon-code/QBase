"""
QBase Strong Trend Strategy v78 — ADX + Composite Regime Score
===============================================================

策略简介：ADX 测量趋势强度，Composite Regime Score 综合判断市场状态，
         双重确认后入场做多。

使用指标：
  - ADX (period=14): > 25 表示存在趋势
  - Composite Regime Score (period=20): 综合趋势 regime 评分
  - ATR (period=14): trailing stop

进场条件：
  1. ADX > 25（趋势存在）
  2. Composite Regime > 0.5（趋势 regime 确认）
  3. 收盘价连续上涨（close > close[-1]）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. ADX < 18（趋势消失）

优点：双重 regime 确认，假信号更少
缺点：进场信号较少，可能错过快速起涨段
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.trend.adx import adx
from indicators.regime.regime_score import composite_regime
from indicators.volatility.atr import atr


class StrongTrendV78(TimeSeriesStrategy):
    """ADX trend strength + Composite Regime Score filter."""
    name = "strong_trend_v78"
    warmup = 60
    freq = "daily"

    adx_period: int = 14
    adx_threshold: float = 25.0
    regime_period: int = 20
    regime_threshold: float = 0.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._adx = None
        self._regime = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
        self.prev_close = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._adx = adx(highs, lows, closes, period=self.adx_period)
        self._regime = composite_regime(closes, highs, lows, period=self.regime_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._closes = closes

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        adx_val = self._adx[i]
        regime_val = self._regime[i]
        atr_val = self._atr[i]
        if np.isnan(adx_val) or np.isnan(regime_val) or np.isnan(atr_val):
            return

        prev_close = self._closes[i - 1] if i > 0 else np.nan
        if np.isnan(prev_close):
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
        if side == 0 and adx_val > self.adx_threshold and regime_val > self.regime_threshold and price > prev_close:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and adx_val < 18.0:
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
