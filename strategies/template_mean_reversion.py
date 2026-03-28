"""
QBase Mean Reversion Strategy Template
=======================================
For oscillation/range-bound markets. Buy oversold, sell overbought.

Key differences from trend templates:
- Fixed profit target (not trailing stop)
- Time-based exit (not trend reversal)
- Tighter stops, smaller position size
- Works both long and short

Exit hierarchy (any can trigger):
  1. Profit target — exit when unrealized profit reaches N × ATR
  2. Stop loss — fixed ATR stop (NOT trailing; MR stops should be tight, 1.5-2.5× ATR)
  3. Time stop — exit after max_holding_bars regardless of PnL

When to use:
- Range-bound / choppy markets (low ADX < 20)
- Instruments with strong mean-reverting properties
- Complement to trend-following strategies for portfolio diversification

Usage:
    ./run.sh strategies/mean_reversion/v<n>.py --symbols <SYMBOL> --freq daily
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

# Import indicators — oscillators for mean reversion
from indicators.momentum.rsi import rsi
from indicators.volatility.atr import atr


class MeanReversionStrategy(TimeSeriesStrategy):
    """
    策略简介：均值回归策略，超卖买入、超买卖出
    使用指标：RSI（超买超卖判断）+ ATR（止损/止盈距离计算）
    进场条件：RSI < oversold_level 做多，RSI > overbought_level 做空
    出场条件：固定止盈（ATR倍数）/ 固定止损（ATR倍数）/ 时间止损（N根K线）
    优点：趋势策略的天然对冲，震荡市表现好
    缺点：趋势市场会反复止损
    """
    name = "mean_reversion_strategy"
    warmup = 60
    freq = "daily"

    # Tunable parameters (≤ 5, narrow ranges)
    rsi_period: int = 14            # RSI lookback
    oversold_level: float = 30.0    # Buy below this RSI
    overbought_level: float = 70.0  # Sell above this RSI
    profit_target_atr: float = 2.0  # Take profit at N × ATR from entry
    stop_loss_atr: float = 2.0      # Fixed stop loss at N × ATR from entry (tight)

    # Non-tunable defaults
    max_holding_bars: int = 20      # Time stop: exit after N bars regardless of PnL

    def __init__(self):
        super().__init__()
        self._rsi = None
        self._atr = None

    def on_init(self, context):
        """Initialize trading state variables."""
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.target_price = 0.0
        self.bars_held = 0
        self.direction = 0  # 1 = long, -1 = short, 0 = flat

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays. Called once."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._rsi = rsi(closes, period=self.rsi_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        """Called on every bar — pure lookup + trading logic."""
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        # V4: Direct property access — do NOT use current_bar attribute
        if context.is_rollover:
            return

        rsi_val = self._rsi[i]
        atr_val = self._atr[i]
        if np.isnan(rsi_val) or np.isnan(atr_val):
            return
        if atr_val <= 0:
            return

        # === Exit Checks (FIRST — survival priority) ===
        if side != 0:
            self.bars_held += 1

            # --- Exit 1: Stop Loss (fixed, NOT trailing) ---
            if side == 1 and price <= self.stop_price:
                context.close_long()
                self._reset()
                return
            if side == -1 and price >= self.stop_price:
                context.close_short()
                self._reset()
                return

            # --- Exit 2: Profit Target ---
            if side == 1 and price >= self.target_price:
                context.close_long()
                self._reset()
                return
            if side == -1 and price <= self.target_price:
                context.close_short()
                self._reset()
                return

            # --- Exit 3: Time Stop ---
            if self.bars_held >= self.max_holding_bars:
                if side == 1:
                    context.close_long()
                else:
                    context.close_short()
                self._reset()
                return

        # === Entry ===
        if side == 0:
            # Long entry: RSI oversold → price likely to bounce up
            if rsi_val < self.oversold_level:
                lot_size = self._calc_lots(context, price, atr_val)
                if lot_size > 0:
                    context.buy(lot_size)
                    self.entry_price = price
                    self.stop_price = price - self.stop_loss_atr * atr_val
                    self.target_price = price + self.profit_target_atr * atr_val
                    self.bars_held = 0
                    self.direction = 1

            # Short entry: RSI overbought → price likely to revert down
            elif rsi_val > self.overbought_level:
                lot_size = self._calc_lots(context, price, atr_val)
                if lot_size > 0:
                    context.sell(lot_size)
                    self.entry_price = price
                    self.stop_price = price + self.stop_loss_atr * atr_val
                    self.target_price = price - self.profit_target_atr * atr_val
                    self.bars_held = 0
                    self.direction = -1

    def _calc_lots(self, context, price, atr_val):
        """Position sizing: risk 1% equity (tighter than trend's 2%), max 30% margin.

        Mean reversion uses smaller size because:
        - Stops are tighter (counter-trend = higher hit rate but smaller wins)
        - We trade both directions, so aggregate exposure can be high
        """
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.stop_loss_atr * atr_val * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.01 / stop_dist)  # 1% risk (not 2%)
        margin = price * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        max_lots = int(context.equity * 0.30 / margin)
        return max(1, min(risk_lots, max_lots))

    def _reset(self):
        """Reset all tracking state when position is closed."""
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.target_price = 0.0
        self.bars_held = 0
        self.direction = 0
