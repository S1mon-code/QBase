"""
QBase Strong Trend Strategy v169 — Model Disagreement (Low = Consensus) + ROC
===============================================================================

策略简介：多模型分歧度量化不同模型预测的一致性，当分歧度低（多模型共识）
         且ROC动量为正时入场做多。

使用指标：
  - Model Disagreement (period=120): 多模型分歧度
  - Rate of Change (period=12): 动量方向
  - ATR (period=14): trailing stop

进场条件：
  1. Model disagreement < 0.3（模型共识度高）
  2. ROC > 0（正向动量）

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. ROC 转负 或 disagreement > 0.7（分歧加大 = 不确定性高）

优点：低分歧 = 高置信度，过滤不确定信号
缺点：高共识时也可能是趋势末期
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
from indicators.ml.disagreement_index import model_disagreement
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr


class StrongTrendV169(TimeSeriesStrategy):
    """模型分歧度共识过滤 + ROC动量方向。"""
    name = "strong_trend_v169"
    warmup = 60
    freq = "daily"

    disagree_period: int = 120
    roc_period: int = 12
    consensus_threshold: float = 0.3
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._disagree = None
        self._roc = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        rsi_arr = rsi(closes, 14)
        adx_arr = adx(highs, lows, closes, 14)
        features = np.column_stack([rsi_arr, adx_arr])

        self._disagree = model_disagreement(closes, features, period=self.disagree_period)
        self._roc = rate_of_change(closes, period=self.roc_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        dis_val = self._disagree[i]
        roc_val = self._roc[i]
        atr_val = self._atr[i]
        if np.isnan(dis_val) or np.isnan(roc_val) or np.isnan(atr_val):
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
        if side == 0 and dis_val < self.consensus_threshold and roc_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and (roc_val < 0 or dis_val > 0.7):
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
