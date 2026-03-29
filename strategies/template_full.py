"""
QBase Full Strategy Template (Pyramiding + Partial Profit-Taking)
=================================================================
Copy this file → strategies/<category>/v<n>.py and modify.

For complex strategies that need:
- Position pyramiding (scale-in up to 3 tiers)
- Partial profit-taking at 3ATR / 5ATR milestones
- Signal-based exit on indicator reversal
- Long + short support (all_time pattern)

For simple strategies (90% of cases): use template_simple.py instead.

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

# Import indicators as needed
from indicators.trend.adx import adx
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr

# =====================================================================
# Position Scaling Configuration
# =====================================================================
SCALE_FACTORS = [1.0, 0.5, 0.25]   # Tier 1 = full, Tier 2 = half, Tier 3 = quarter
MAX_SCALE = 3


class FullStrategy(TimeSeriesStrategy):
    """
    策略简介：[一句话描述]
    使用指标：[列出指标及作用]
    进场条件：[做多/做空触发条件]
    出场条件：[止损/分层止盈/信号反转]
    优点：[核心优势]
    缺点：[已知局限]
    """
    name = "full_strategy"
    warmup = 60
    freq = "daily"

    # ----- Tunable parameters (≤ 5) -----
    adx_period: int = 14
    roc_period: int = 20
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._adx = None
        self._roc = None
        self._atr = None

    # =====================================================================
    # Initialization
    # =====================================================================

    def on_init(self, context):
        """Initialize trading state variables."""
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.direction = 0  # 1 = long, -1 = short, 0 = flat

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays. Called once."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._adx = adx(highs, lows, closes, period=self.adx_period)
        self._roc = rate_of_change(closes, self.roc_period)
        self._atr = atr(highs, lows, closes, period=14)

    # =====================================================================
    # Core Bar Handler
    # =====================================================================

    def on_bar(self, context):
        """Called on every bar — pure lookup + trading logic."""
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        # V4: Direct property access — do NOT use current_bar attribute
        if context.is_rollover:
            return

        adx_val = self._adx[i]
        roc_val = self._roc[i]
        atr_val = self._atr[i]

        if np.isnan(adx_val) or np.isnan(roc_val) or np.isnan(atr_val):
            return
        if atr_val <= 0:
            return

        self.bars_since_last_scale += 1

        # === 1. STOP LOSS (survival priority) ===
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return
        elif side == -1:
            self.lowest_since_entry = min(self.lowest_since_entry, price)
            trailing = self.lowest_since_entry + self.atr_trail_mult * atr_val
            self.stop_price = min(self.stop_price, trailing)
            if price >= self.stop_price:
                context.close_short()
                self._reset_state()
                return

        # === 2. TIERED PROFIT-TAKING ===
        if side != 0 and self.entry_price > 0:
            if side == 1:
                profit_atr = (price - self.entry_price) / atr_val
            else:
                profit_atr = (self.entry_price - price) / atr_val

            if profit_atr >= 5.0 and not self._took_profit_5atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_3atr = True
                return

        # === 3. SIGNAL-BASED EXIT ===
        if side == 1 and adx_val < 20:  # Customize: trend weakens
            context.close_long()
            self._reset_state()
            return
        if side == -1 and adx_val < 20:  # Customize: trend weakens
            context.close_short()
            self._reset_state()
            return

        # === 4. ENTRY ===
        if side == 0:
            # Long entry
            if adx_val > 25 and roc_val > 0:  # Customize conditions
                base_lots = self._calc_lots(context, price, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_trail_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
                    self.direction = 1

            # Short entry (for all_time strategies)
            elif adx_val > 25 and roc_val < 0:  # Customize conditions
                base_lots = self._calc_lots(context, price, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_trail_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
                    self.direction = -1

            # For long-only (strong_trend): remove the short entry block above

        # === 5. SCALE-IN (pyramiding) ===
        elif side != 0 and self._should_add(price, atr_val):
            if self.direction == 1:
                add_lots = self._calc_add_lots(self._calc_lots(context, price, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1:
                add_lots = self._calc_add_lots(self._calc_lots(context, price, atr_val))
                if add_lots > 0:
                    context.sell(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0

    # =====================================================================
    # Scale-In Logic
    # =====================================================================

    def _should_add(self, price, atr_val):
        """Check if conditions are met for adding to position."""
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        # Only add if price has moved favorably by at least 1 ATR
        if self.direction == 1 and price < self.entry_price + atr_val:
            return False
        if self.direction == -1 and price > self.entry_price - atr_val:
            return False
        return True

    def _calc_add_lots(self, base_lots):
        """Scale down lot size for each subsequent tier."""
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    # =====================================================================
    # Position Sizing
    # =====================================================================

    def _calc_lots(self, context, price, atr_val):
        """Position sizing: risk 2% equity, max 30% margin.

        Uses module-level _SPEC_MANAGER (V4 correct pattern).
        """
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

    # =====================================================================
    # State Reset
    # =====================================================================

    def _reset_state(self):
        """Reset all tracking state when position is fully closed."""
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.direction = 0
