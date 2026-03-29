"""
QBase Simple Strategy Template
===============================
Copy this file → strategies/<category>/v<n>.py and modify.

For 90% of strategies: 1-2 indicators + ATR trailing stop.
For complex strategies (pyramiding, multi-TF): use template_full.py.

Usage:
    ./run.sh strategies/<category>/v<n>.py --symbols <SYMBOL> --freq daily --start 2022

Naming convention:
    strategies/strong_trend/v51.py  — 强趋势策略 v51
    strategies/all_time/ag/v201.py  — 白银全时间策略 v201
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

# V4: Module-level singleton — only reads YAML once
_SPEC_MANAGER = ContractSpecManager()

# Import indicators as needed
from indicators.volume.volume_momentum import volume_momentum
from indicators.volatility.atr import atr


class SimpleStrategy(TimeSeriesStrategy):
    """
    策略简介：[一句话描述]
    使用指标：[列出指标及作用]
    进场条件：[做多/做空触发条件]
    出场条件：[止损/信号反转条件]
    优点：[核心优势]
    缺点：[已知局限]
    """
    name = "simple_strategy"
    warmup = 60
    freq = "daily"

    # Tunable parameters (≤ 5, narrow ranges)
    indicator_period: int = 14
    atr_trail_mult: float = 4.5  # Wide stop for trend strategies

    def __init__(self):
        super().__init__()
        self._indicator = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._indicator = volume_momentum(context.get_full_volume_array(), self.indicator_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        # V4: Direct property access — do NOT use current_bar attribute
        if context.is_rollover:
            return

        ind_val = self._indicator[i]
        atr_val = self._atr[i]
        if np.isnan(ind_val) or np.isnan(atr_val):
            return

        # === Stop Loss Check (FIRST — survival priority) ===
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # === Entry ===
        if side == 0 and ind_val > 2.0:  # Customize condition
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and ind_val < 0.5:  # Customize condition
            context.close_long()
            self._reset()

        # === For all_time (long+short): uncomment below ===
        # --- Short-side stop loss ---
        # if side == -1:
        #     self.lowest_since_entry = min(self.lowest_since_entry, price)
        #     trailing = self.lowest_since_entry + self.atr_trail_mult * atr_val
        #     self.stop_price = min(self.stop_price, trailing)
        #     if price >= self.stop_price:
        #         context.close_short()
        #         self._reset()
        #         return
        #
        # --- Short entry ---
        # elif side == 0 and ind_val < -2.0:
        #     lot_size = self._calc_lots(context, price, atr_val)
        #     if lot_size > 0:
        #         context.sell(lot_size)
        #         self.entry_price = price
        #         self.stop_price = price + self.atr_trail_mult * atr_val
        #         self.lowest_since_entry = price
        #
        # --- Short signal exit ---
        # elif side == -1 and ind_val > -0.5:
        #     context.close_short()
        #     self._reset()

    def _calc_lots(self, context, price, atr_val):
        """Position sizing: risk 2% equity, max 30% margin."""
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
