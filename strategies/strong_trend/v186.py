"""
QBase Strong Trend Strategy v186 — Sharpe-like Signal (Momentum/Volatility) + McGinley
========================================================================================

策略简介：构建类Sharpe信号（ROC/滚动波动率），当风险调整后动量为正且McGinley动态均线
         确认趋势时做多。高Sharpe信号 = 高质量动量。

使用指标：
  - ROC (period=20) / Historical Vol (period=20): Sharpe-like ratio
  - McGinley Dynamic (period=14): 自适应趋势线
  - ATR (period=14): trailing stop

进场条件：
  1. Sharpe-like signal > 1.0（风险调整动量强）
  2. 价格 > McGinley动态均线（趋势向上）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. Sharpe-like signal < 0（风险调整动量转负）

优点：质量动量 + 自适应趋势确认
缺点：短窗口Sharpe估计噪声大
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.momentum.roc import rate_of_change
from indicators.volatility.historical_vol import historical_volatility
from indicators.trend.mcginley import mcginley_dynamic
from indicators.volatility.atr import atr


class StrongTrendV186(TimeSeriesStrategy):
    """Sharpe-like信号(动量/波动率) + McGinley趋势策略。"""
    name = "strong_trend_v186"
    warmup = 60
    freq = "daily"

    mom_period: int = 20
    mcg_period: int = 14
    sharpe_entry: float = 1.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._sharpe = None
        self._mcg = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        roc_vals = rate_of_change(closes, period=self.mom_period)
        hvol = historical_volatility(closes, period=self.mom_period, ann=252)
        # Sharpe-like = ROC / vol, avoid div by zero
        self._sharpe = np.where(hvol > 1e-10, roc_vals / hvol, 0.0)
        self._mcg = mcginley_dynamic(closes, period=self.mcg_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        sh = self._sharpe[i]
        mcg_val = self._mcg[i]
        atr_val = self._atr[i]
        if np.isnan(sh) or np.isnan(mcg_val) or np.isnan(atr_val):
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
        if side == 0 and sh > self.sharpe_entry and price > mcg_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and sh < 0.0:
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
