"""
QBase Strong Trend Strategy v170 — Feature Importance (Tree-Based) + Volume Momentum
======================================================================================

策略简介：滚动树模型特征重要性评估当前市场可预测性，当特征重要性高
         且成交量动量确认放量时入场做多。

使用指标：
  - Feature Importance (period=120, n_estimators=50): 树模型特征重要性
  - Volume Momentum (period=14): 成交量动量
  - ATR (period=14): trailing stop

进场条件：
  1. 最大特征重要性 > 0.3（市场存在可预测模式）
  2. Volume Momentum > 1.0（放量确认）
  3. 收盘价高于20日均线（基本趋势向上）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Volume Momentum < 0.5（缩量）

优点：特征重要性反映市场可预测性，高时信号更可靠
缺点：树模型对样本量敏感
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
from indicators.ml.feature_importance import rolling_tree_importance
from indicators.volume.volume_momentum import volume_momentum
from indicators.trend.sma import sma
from indicators.volatility.atr import atr


class StrongTrendV170(TimeSeriesStrategy):
    """树模型特征重要性 + 成交量动量确认。"""
    name = "strong_trend_v170"
    warmup = 60
    freq = "daily"

    fi_period: int = 120
    vol_mom_period: int = 14
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._importance = None
        self._vol_mom = None
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
        volumes = context.get_full_volume_array()

        rsi_arr = rsi(closes, 14)
        adx_arr = adx(highs, lows, closes, 14)
        features = np.column_stack([rsi_arr, adx_arr])

        importance = rolling_tree_importance(closes, features, period=self.fi_period, n_estimators=50)
        # Max importance across features per bar
        if importance.ndim > 1:
            self._importance = np.nanmax(importance, axis=1)
        else:
            self._importance = importance
        self._vol_mom = volume_momentum(volumes, period=self.vol_mom_period)
        self._sma20 = sma(closes, 20)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        imp_val = self._importance[i]
        vm_val = self._vol_mom[i]
        sma_val = self._sma20[i]
        atr_val = self._atr[i]
        if np.isnan(imp_val) or np.isnan(vm_val) or np.isnan(sma_val) or np.isnan(atr_val):
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
        if side == 0 and imp_val > 0.3 and vm_val > 1.0 and price > sma_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and vm_val < 0.5:
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
