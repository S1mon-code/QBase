"""
QBase Strong Trend Strategy v197 — TSI (True Strength Index) + Chandelier Exit
=================================================================================

策略简介：TSI是双重平滑动量指标，结合Chandelier Exit作为动态止损和趋势确认。
         TSI上穿信号线 + 价格在Chandelier Long Exit上方 = 强趋势入场。

使用指标：
  - TSI (long=25, short=13, signal=7): 真实强度指数
  - Chandelier Exit (period=22, mult=3.0): 动态止损线
  - ATR (period=14): trailing stop

进场条件：
  1. TSI > TSI Signal（动量上穿信号线）
  2. 价格 > Chandelier Long Exit（在趋势止损线上方）

出场条件：
  1. ATR trailing stop (mult=4.0)
  2. TSI < TSI Signal 且 TSI < 0（动量反转）

优点：TSI平滑可靠 + Chandelier动态止损
缺点：双重平滑导致入场稍滞后
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.momentum.tsi import tsi
from indicators.volatility.chandelier_exit import chandelier_exit
from indicators.volatility.atr import atr


class StrongTrendV197(TimeSeriesStrategy):
    """TSI真实强度 + Chandelier Exit策略。"""
    name = "strong_trend_v197"
    warmup = 60
    freq = "daily"

    tsi_long: int = 25
    tsi_short: int = 13
    ch_period: int = 22
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._tsi = None
        self._tsi_sig = None
        self._ch_long = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._tsi, self._tsi_sig = tsi(closes, long=self.tsi_long, short=self.tsi_short, signal=7)
        self._ch_long, _ = chandelier_exit(highs, lows, closes,
                                            period=self.ch_period, mult=3.0)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        tsi_val = self._tsi[i]
        tsi_s = self._tsi_sig[i]
        ch_l = self._ch_long[i]
        atr_val = self._atr[i]
        if np.isnan(tsi_val) or np.isnan(tsi_s) or np.isnan(ch_l) or np.isnan(atr_val):
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
        if side == 0 and tsi_val > tsi_s and price > ch_l:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and tsi_val < tsi_s and tsi_val < 0:
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
