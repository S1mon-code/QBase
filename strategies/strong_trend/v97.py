"""
QBase Strong Trend Strategy v97 — Distribution Shift (KL) + Chandelier Exit
==============================================================================

策略简介：KL Divergence 检测收益率分布偏移（regime 变化前兆），
         Chandelier Exit 提供动态止损和方向确认。

使用指标：
  - Distribution Shift / KL Divergence (period=60, ref=120): 分布变化检测
  - Chandelier Exit (period=22, mult=3.0): 动态止损线
  - ATR (period=14): trailing stop

进场条件：
  1. KL Divergence < 0.5（分布稳定 = 当前 regime 持续）
  2. 收盘价 > Chandelier Long line（价格在上方趋势支撑上）
  3. 收盘价 > 前一日收盘价

出场条件：
  1. ATR trailing stop（mult=4.0）
  2. KL Divergence > 2.0（分布剧烈变化 = regime 切换）

优点：KL Divergence 从分布角度检测 regime 变化，信息量丰富
缺点：KL Divergence 对 bins 选择敏感
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.distribution_shift import kl_divergence_shift
from indicators.volatility.chandelier_exit import chandelier_exit
from indicators.volatility.atr import atr


class StrongTrendV97(TimeSeriesStrategy):
    """KL Distribution Shift regime + Chandelier Exit direction."""
    name = "strong_trend_v97"
    warmup = 60
    freq = "daily"

    kl_stable_thresh: float = 0.5
    kl_shift_thresh: float = 2.0
    chand_period: int = 22
    chand_mult: float = 3.0
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._kl = None
        self._chand_long = None
        self._chand_short = None
        self._atr = None
        self._closes = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._kl = kl_divergence_shift(closes, period=60, reference_period=120)
        self._chand_long, self._chand_short = chandelier_exit(
            highs, lows, closes, period=self.chand_period, mult=self.chand_mult
        )
        self._atr = atr(highs, lows, closes, period=14)
        self._closes = closes

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        kl_val = self._kl[i]
        cl = self._chand_long[i]
        atr_val = self._atr[i]
        if np.isnan(kl_val) or np.isnan(cl) or np.isnan(atr_val) or i < 1:
            return

        prev_close = self._closes[i - 1]

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
        if side == 0 and kl_val < self.kl_stable_thresh and price > cl and price > prev_close:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and kl_val > self.kl_shift_thresh:
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
