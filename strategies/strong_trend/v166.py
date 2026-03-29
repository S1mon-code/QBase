"""
QBase Strong Trend Strategy v166 — Attention Score + OI Flow
==============================================================

策略简介：注意力权重机制对特征进行动态加权，生成趋势信号，
         配合OI Flow确认资金流方向后入场。

使用指标：
  - Attention Score (period=60): 注意力加权特征信号
  - OI Flow (period=14): 持仓量资金流向
  - ATR (period=14): trailing stop

进场条件：
  1. Attention score > 0（注意力加权信号看多）
  2. OI Flow > 0（资金净流入）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. Attention score < 0（信号翻转）

优点：动态注意力权重自适应重要特征
缺点：需要合理的目标变量构建
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
from indicators.ml.attention_score import attention_weights
from indicators.volume.oi_flow import oi_flow
from indicators.volatility.atr import atr


class StrongTrendV166(TimeSeriesStrategy):
    """注意力权重趋势信号 + OI Flow资金流确认。"""
    name = "strong_trend_v166"
    warmup = 60
    freq = "daily"

    attn_period: int = 60
    oi_flow_period: int = 14
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._attn = None
        self._oi_flow = None
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
        oi = context.get_full_oi_array()

        rsi_arr = rsi(closes, 14)
        adx_arr = adx(highs, lows, closes, 14)
        features = np.column_stack([rsi_arr, adx_arr])

        # Target: forward returns
        target = np.full(len(closes), np.nan)
        target[:-5] = (closes[5:] - closes[:-5]) / closes[:-5]

        self._attn = attention_weights(features, target, period=self.attn_period)
        self._oi_flow = oi_flow(closes, oi, volumes, period=self.oi_flow_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        attn_val = self._attn[i] if self._attn.ndim == 1 else self._attn[i, 0]
        oif_val = self._oi_flow[i]
        atr_val = self._atr[i]
        if np.isnan(attn_val) or np.isnan(oif_val) or np.isnan(atr_val):
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
        if side == 0 and attn_val > 0 and oif_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and attn_val < 0:
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
