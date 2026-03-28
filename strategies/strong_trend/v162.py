"""
QBase Strong Trend Strategy v162 — Ensemble Vote (Multiple Models) + OBV
=========================================================================

策略简介：集成多个模型投票产生共识信号，当多数模型看多且OBV确认
         资金流入时入场做多。

使用指标：
  - Ensemble Vote (period=120): 多模型集成投票信号
  - OBV: 能量潮指标（资金流方向）
  - ATR (period=14): trailing stop

进场条件：
  1. Ensemble vote > 0.5（多数模型看多）
  2. OBV上升（当前 > 20日前）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Ensemble vote < 0（多数模型看空）

优点：集成方法降低单一模型过拟合风险
缺点：计算开销较大
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx
from indicators.ml.ensemble_signal import ensemble_vote
from indicators.volume.obv import obv
from indicators.volatility.atr import atr


class StrongTrendV162(TimeSeriesStrategy):
    """多模型集成投票 + OBV资金流确认。"""
    name = "strong_trend_v162"
    warmup = 60
    freq = "daily"

    ensemble_period: int = 120
    obv_lookback: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._ensemble = None
        self._obv = None
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

        rsi_arr = rsi(closes, 14)
        adx_arr = adx(highs, lows, closes, 14)
        features = np.column_stack([rsi_arr, adx_arr])

        self._ensemble = ensemble_vote(closes, features, period=self.ensemble_period)
        self._obv = obv(closes, volumes)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        ens_val = self._ensemble[i]
        obv_val = self._obv[i]
        obv_prev = self._obv[i - self.obv_lookback] if i >= self.obv_lookback else np.nan
        atr_val = self._atr[i]
        if np.isnan(ens_val) or np.isnan(obv_val) or np.isnan(obv_prev) or np.isnan(atr_val):
            return

        obv_rising = obv_val > obv_prev

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
        if side == 0 and ens_val > 0.5 and obv_rising:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and ens_val < 0:
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
