"""
QBase Strong Trend Strategy v182 — Rolling Beta (vs own lagged returns) + TEMA
================================================================================

策略简介：用收益率与自身滞后收益率的滚动Beta衡量动量持续性，结合TEMA确认趋势方向。
         Beta>1表示动量自我强化。

使用指标：
  - Rolling Beta (period=60): 收益率vs滞后收益率的beta
  - TEMA (period=20): 三重指数移动平均趋势
  - ATR (period=14): trailing stop

进场条件：
  1. Rolling Beta > 1.0（动量自我强化）
  2. 价格 > TEMA（趋势向上）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. 价格 < TEMA（趋势翻转）

优点：Beta度量动量持续性，TEMA平滑确认
缺点：Beta计算需要较长窗口
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.spread.beta import rolling_beta
from indicators.trend.tema import tema
from indicators.volatility.atr import atr


class StrongTrendV182(TimeSeriesStrategy):
    """滚动Beta(动量持续性) + TEMA趋势策略。"""
    name = "strong_trend_v182"
    warmup = 60
    freq = "daily"

    beta_period: int = 60
    tema_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._beta = None
        self._tema = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        # Compute returns and lagged returns as proxy for self-momentum
        returns = np.full(len(closes), np.nan)
        returns[1:] = closes[1:] / closes[:-1] - 1.0
        lagged_returns = np.full(len(closes), np.nan)
        lagged_returns[2:] = returns[1:-1]

        self._beta, _, _ = rolling_beta(returns, lagged_returns, period=self.beta_period)
        self._tema = tema(closes, period=self.tema_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        beta_val = self._beta[i]
        tema_val = self._tema[i]
        atr_val = self._atr[i]
        if np.isnan(beta_val) or np.isnan(tema_val) or np.isnan(atr_val):
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
        if side == 0 and beta_val > 1.0 and price > tema_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and price < tema_val:
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
