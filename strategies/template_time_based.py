"""
QBase Time-Based Exit Strategy Template
========================================
Enters on signal, exits after fixed number of bars.
No trailing stop — pure time exit.

Use case: momentum strategies where you want to capture the initial move
and exit before mean reversion kicks in.

Exit mechanism:
- Primary: exit after hold_bars bars (configurable, e.g., 5, 10, 20)
- Emergency: fixed percentage stop as safety net (e.g., -5% from entry)
- No trailing stop at all

When to use:
- Event-driven / momentum-burst strategies
- When holding too long causes alpha decay (e.g., earnings drift)
- Simpler than ATR templates — fewer parameters, less curve-fitting risk

Usage:
    ./run.sh strategies/<category>/v<n>.py --symbols <SYMBOL> --freq daily --start 2022
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

# Import indicators — signal indicator is customizable
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr


class TimeBasedStrategy(TimeSeriesStrategy):
    """
    策略简介：时间退出策略，固定持有N根K线后无条件退出
    使用指标：ROC（进场信号，可替换）+ ATR（仓位计算）
    进场条件：信号触发（可自定义）
    出场条件：持有N根K线后退出 / 紧急百分比止损
    优点：简单、参数少、避免过度持有导致alpha衰减
    缺点：可能过早退出趋势行情，需要配合信号质量
    """
    name = "time_based_strategy"
    warmup = 60
    freq = "daily"

    # Tunable parameters (≤ 5, narrow ranges)
    signal_period: int = 20         # Signal indicator lookback
    signal_threshold: float = 3.0   # Min signal strength to enter
    hold_bars: int = 10             # Exit after N bars (core parameter)
    emergency_stop_pct: float = 0.05  # Emergency stop: -5% from entry

    def __init__(self):
        super().__init__()
        self._signal = None
        self._atr = None

    def on_init(self, context):
        """Initialize trading state variables."""
        self.entry_price = 0.0
        self.bars_held = 0
        self.direction = 0  # 1 = long, -1 = short, 0 = flat

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays. Called once."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        # Signal: ROC as default (swap for any indicator)
        self._signal = rate_of_change(closes, self.signal_period)

        # ATR for position sizing only (NOT for stops)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        """Called on every bar — pure lookup + trading logic."""
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        # V4: Direct property access — do NOT use current_bar attribute
        if context.is_rollover:
            return

        signal_val = self._signal[i]
        atr_val = self._atr[i]
        if np.isnan(signal_val) or np.isnan(atr_val):
            return
        if atr_val <= 0:
            return

        # === Exit Checks (FIRST — survival priority) ===
        if side != 0:
            self.bars_held += 1

            # --- Exit 1: Emergency Percentage Stop ---
            if side == 1 and price <= self.entry_price * (1.0 - self.emergency_stop_pct):
                context.close_long()
                self._reset()
                return
            if side == -1 and price >= self.entry_price * (1.0 + self.emergency_stop_pct):
                context.close_short()
                self._reset()
                return

            # --- Exit 2: Time Exit (core mechanism) ---
            if self.bars_held >= self.hold_bars:
                if side == 1:
                    context.close_long()
                else:
                    context.close_short()
                self._reset()
                return

        # === Entry ===
        if side == 0:
            # Long entry
            if signal_val > self.signal_threshold:
                lot_size = self._calc_lots(context, price, atr_val)
                if lot_size > 0:
                    context.buy(lot_size)
                    self.entry_price = price
                    self.bars_held = 0
                    self.direction = 1

            # Short entry
            elif signal_val < -self.signal_threshold:
                lot_size = self._calc_lots(context, price, atr_val)
                if lot_size > 0:
                    context.sell(lot_size)
                    self.entry_price = price
                    self.bars_held = 0
                    self.direction = -1

    def _calc_lots(self, context, price, atr_val):
        """Position sizing: risk 2% equity based on emergency stop distance, max 30% margin."""
        spec = _SPEC_MANAGER.get(context.symbol)
        # Use emergency stop distance for sizing (since that's our actual risk)
        stop_dist = self.emergency_stop_pct * price * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = price * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        max_lots = int(context.equity * 0.30 / margin)
        return max(1, min(risk_lots, max_lots))

    def _reset(self):
        """Reset all tracking state when position is closed."""
        self.entry_price = 0.0
        self.bars_held = 0
        self.direction = 0
