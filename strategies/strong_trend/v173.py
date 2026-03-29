"""
QBase Strong Trend Strategy v173 — Symbolic Regression Features + Force Index
===============================================================================

策略简介：符号回归自动发现价格的数学表达式特征，当发现的特征信号
         为正且Force Index确认买压时入场做多。

使用指标：
  - Symbolic Regression Features (period=60): 自动发现数学特征
  - Force Index (period=13): 量价合力
  - ATR (period=14): trailing stop

进场条件：
  1. Symbolic features主信号 > 0（发现的表达式看多）
  2. Force Index > 0（买压大于卖压）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Symbolic features主信号 < 0

优点：自动特征工程，无需人工设计指标
缺点：发现的表达式可能过拟合
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.ml.symbolic_regression_signal import symbolic_features
from indicators.volume.force_index import force_index
from indicators.volatility.atr import atr


class StrongTrendV173(TimeSeriesStrategy):
    """符号回归自动特征 + Force Index量价确认。"""
    name = "strong_trend_v173"
    warmup = 60
    freq = "daily"

    sym_period: int = 60
    force_period: int = 13
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._sym_signal = None
        self._force = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        sym = symbolic_features(closes, period=self.sym_period)
        # First column is the primary signal
        self._sym_signal = sym[:, 0] if sym.ndim > 1 else sym
        self._force = force_index(closes, volumes, period=self.force_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        sym_val = self._sym_signal[i]
        force_val = self._force[i]
        atr_val = self._atr[i]
        if np.isnan(sym_val) or np.isnan(force_val) or np.isnan(atr_val):
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
        if side == 0 and sym_val > 0 and force_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and sym_val < 0:
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
