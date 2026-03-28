"""
Strong Trend Strategy v108 — Chandelier Exit + Yang-Zhang Vol Expansion
=========================================================================
Uses Chandelier Exit for trend-following stop management and Yang-Zhang
volatility to detect expansion from low-volatility regimes.

  1. Chandelier Exit    — ATR-based trailing stop from highest high
  2. Yang-Zhang Vol     — efficient OHLC volatility expansion detection

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v108.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.chandelier_exit import chandelier_exit
from indicators.volatility.yang_zhang import yang_zhang
from indicators.volatility.atr import atr


class StrongTrendV108(TimeSeriesStrategy):
    """
    策略简介：Chandelier Exit + Yang-Zhang波动率扩张的趋势跟踪策略
    使用指标：Chandelier Exit（趋势止损）、Yang-Zhang Vol（波动率状态）
    进场条件：YZ Vol从低位扩张 + 价格在Chandelier Long Exit上方
    出场条件：ATR trailing stop 或 价格跌破Chandelier Long Exit
    优点：Chandelier Exit是经典趋势止损，YZ Vol效率高于Close-to-Close
    缺点：双重波动率指标可能在低波市场长期空仓
    """
    name = "strong_trend_v108"
    warmup = 60
    freq = "daily"

    chand_period: int = 22
    chand_mult: float = 3.0
    yz_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._chand_long = None
        self._chand_short = None
        self._yz = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        opens = context.get_full_open_array()

        self._chand_long, self._chand_short = chandelier_exit(
            highs, lows, closes, period=self.chand_period, multiplier=self.chand_mult
        )
        self._yz = yang_zhang(opens, highs, lows, closes, period=self.yz_period)
        self._atr = atr(highs, lows, closes, period=14)

        # Pre-compute YZ vol rolling average
        n = len(closes)
        self._yz_avg = np.full(n, np.nan)
        for k in range(39, n):
            window = self._yz[k - 39:k + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                self._yz_avg[k] = np.mean(valid)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        chand_long = self._chand_long[i]
        cur_yz = self._yz[i]
        yz_avg = self._yz_avg[i]
        atr_val = self._atr[i]

        if np.isnan(chand_long) or np.isnan(cur_yz) or np.isnan(yz_avg) or np.isnan(atr_val):
            return

        yz_expanding = cur_yz > 1.3 * yz_avg
        above_chandelier = price > chand_long

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
        if side == 0 and yz_expanding and above_chandelier:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and price < chand_long:
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
