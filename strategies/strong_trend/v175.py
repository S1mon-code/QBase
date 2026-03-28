"""
QBase Strong Trend Strategy v175 — Von Neumann Ratio (Trend Detection) + OI Momentum
======================================================================================

策略简介：Von Neumann比率检测序列中的趋势性（比率<2表示正自相关/趋势），
         当检测到趋势且OI动量确认资金流入时入场做多。

使用指标：
  - Von Neumann Ratio (period=60): 连续差分比率（<2=趋势, >2=均值回归）
  - OI Momentum (period=20): 持仓量动量
  - ATR (period=14): trailing stop

进场条件：
  1. Von Neumann ratio < 1.5（强趋势特征）
  2. OI Momentum > 0（持仓量增加，资金流入）
  3. 收盘价高于20日均线（确认趋势方向为上）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Von Neumann ratio > 2.0（趋势消失）

优点：经典统计检验，理论基础扎实
缺点：对周期长度敏感
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.ml.successive_differences import von_neumann_ratio
from indicators.volume.oi_momentum import oi_momentum
from indicators.trend.sma import sma
from indicators.volatility.atr import atr


class StrongTrendV175(TimeSeriesStrategy):
    """Von Neumann比率趋势检测 + OI动量资金流确认。"""
    name = "strong_trend_v175"
    warmup = 60
    freq = "daily"

    vn_period: int = 60
    oi_mom_period: int = 20
    vn_threshold: float = 1.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._vn = None
        self._oi_mom = None
        self._sma20 = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        oi = context.get_full_oi_array()

        self._vn = von_neumann_ratio(closes, period=self.vn_period)
        self._oi_mom = oi_momentum(oi, period=self.oi_mom_period)
        self._sma20 = sma(closes, 20)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        vn_val = self._vn[i]
        oi_val = self._oi_mom[i]
        sma_val = self._sma20[i]
        atr_val = self._atr[i]
        if np.isnan(vn_val) or np.isnan(oi_val) or np.isnan(sma_val) or np.isnan(atr_val):
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
        if side == 0 and vn_val < self.vn_threshold and oi_val > 0 and price > sma_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and vn_val > 2.0:
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
