"""
QBase Volatility Target Strategy Template
==========================================
Uses volatility targeting for position sizing and risk management.
Instead of ATR trailing stop, adjusts position size inversely to volatility.

Key concept:
- High volatility → smaller position → same dollar risk
- Low volatility → larger position → same dollar risk
- Target annualized volatility: 15% (configurable)

Exit mechanism:
- Percentage stop (-3% from entry) instead of ATR trailing stop
- Signal-based exit (customizable)

When to use:
- Strategies where consistent portfolio-level risk is priority
- Instruments with highly variable volatility (e.g., commodities)
- When you want to avoid being over-sized in high-vol and under-sized in low-vol

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

# Import indicators — signal indicator is customizable, vol is core
from indicators.volatility.historical_vol import historical_volatility
from indicators.momentum.roc import rate_of_change


class VolatilityTargetStrategy(TimeSeriesStrategy):
    """
    策略简介：波动率目标策略，根据实现波动率动态调仓
    使用指标：历史波动率（仓位计算）+ ROC（信号，可替换）
    进场条件：信号触发时根据波动率目标计算仓位
    出场条件：百分比止损（-3%）/ 信号反转
    优点：风险恒定，高波缩仓低波加仓，收益更平滑
    缺点：低波时杠杆可能过高，需要额外杠杆上限
    """
    name = "volatility_target_strategy"
    warmup = 60
    freq = "daily"

    # Tunable parameters (≤ 5, narrow ranges)
    vol_target: float = 0.15        # Target annualized volatility (15%)
    vol_lookback: int = 20          # Rolling window for realized vol
    signal_period: int = 20         # Signal indicator lookback (ROC default)
    pct_stop: float = 0.03          # Percentage stop loss from entry (3%)
    max_leverage: float = 3.0       # Max leverage cap to prevent blow-up in low-vol

    def __init__(self):
        super().__init__()
        self._realized_vol = None
        self._signal = None

    def on_init(self, context):
        """Initialize trading state variables."""
        self.entry_price = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays. Called once."""
        closes = context.get_full_close_array()

        # Core: realized volatility for position sizing
        self._realized_vol = historical_volatility(
            closes, period=self.vol_lookback, ann=252
        )

        # Signal: ROC as default (swap for any indicator)
        self._signal = rate_of_change(closes, self.signal_period)

    def on_bar(self, context):
        """Called on every bar — pure lookup + trading logic."""
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        # V4: Direct property access — do NOT use current_bar attribute
        if context.is_rollover:
            return

        vol_val = self._realized_vol[i]
        signal_val = self._signal[i]
        if np.isnan(vol_val) or np.isnan(signal_val):
            return
        if vol_val <= 0:
            return

        # === Stop Loss Check (FIRST — survival priority) ===
        if side == 1:
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return
        elif side == -1:
            if price >= self.stop_price:
                context.close_short()
                self._reset()
                return

        # === Signal Exit ===
        if side == 1 and signal_val < 0:  # Customize: signal turns negative
            context.close_long()
            self._reset()
            return
        if side == -1 and signal_val > 0:  # Customize: signal turns positive
            context.close_short()
            self._reset()
            return

        # === Entry ===
        if side == 0:
            # Long entry
            if signal_val > 2.0:  # Customize condition
                lot_size = self._calc_lots_vol_target(context, price, vol_val)
                if lot_size > 0:
                    context.buy(lot_size)
                    self.entry_price = price
                    self.stop_price = price * (1.0 - self.pct_stop)

            # Short entry
            elif signal_val < -2.0:  # Customize condition
                lot_size = self._calc_lots_vol_target(context, price, vol_val)
                if lot_size > 0:
                    context.sell(lot_size)
                    self.entry_price = price
                    self.stop_price = price * (1.0 + self.pct_stop)

    def _calc_lots_vol_target(self, context, price, realized_vol):
        """Volatility-targeted position sizing.

        Core formula:
            target_exposure = equity * (vol_target / realized_vol)
            lots = target_exposure / (price * multiplier)

        Capped by max_leverage to prevent blow-up in low-vol regimes.
        """
        spec = _SPEC_MANAGER.get(context.symbol)
        notional_per_lot = price * spec.multiplier
        if notional_per_lot <= 0:
            return 0

        # Vol-target sizing: scale inversely to realized vol
        vol_scalar = self.vol_target / realized_vol
        vol_scalar = min(vol_scalar, self.max_leverage)  # Cap leverage

        target_exposure = context.equity * vol_scalar
        vol_lots = int(target_exposure / notional_per_lot)

        # Margin constraint
        margin = notional_per_lot * spec.margin_rate
        if margin <= 0:
            return 0
        max_lots = int(context.equity * 0.50 / margin)  # Higher margin for vol-target

        return max(1, min(vol_lots, max_lots))

    def _reset(self):
        """Reset all tracking state when position is closed."""
        self.entry_price = 0.0
        self.stop_price = 0.0
