"""
QBase Strong Trend Strategy v89 — Stationarity Score + HMA
============================================================

策略简介：Stationarity Score 低值表示价格序列非平稳（趋势特征），
         HMA 提供低延迟的趋势方向确认。

使用指标：
  - Stationarity Score (period=60): 低值 = 非平稳 = 趋势
  - HMA (period=20): Hull Moving Average，低延迟均线
  - ATR (period=14): trailing stop

进场条件：
  1. Stationarity < 0.3（非平稳 = 趋势 regime）
  2. 收盘价 > HMA（趋势向上）
  3. HMA 斜率 > 0（方向确认）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Stationarity > 0.7（平稳 = 均值回归 regime）

优点：Stationarity 直接检验时序是否有单位根，HMA 延迟极低
缺点：ADF 检验在小样本中 power 不足
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.stationarity_score import stationarity
from indicators.trend.hma import hma
from indicators.volatility.atr import atr


class StrongTrendV89(TimeSeriesStrategy):
    """Low Stationarity Score (trending) + HMA direction."""
    name = "strong_trend_v89"
    warmup = 60
    freq = "daily"

    stat_threshold: float = 0.3
    hma_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._stat = None
        self._hma = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._stat = stationarity(closes, period=60)
        self._hma = hma(closes, period=self.hma_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        stat_val = self._stat[i]
        hma_val = self._hma[i]
        atr_val = self._atr[i]
        if np.isnan(stat_val) or np.isnan(hma_val) or np.isnan(atr_val) or i < 1:
            return

        hma_prev = self._hma[i - 1]
        if np.isnan(hma_prev):
            return

        hma_slope = hma_val - hma_prev

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
        if side == 0 and stat_val < self.stat_threshold and price > hma_val and hma_slope > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and stat_val > 0.7:
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
