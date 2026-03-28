"""
QBase Strong Trend Strategy v183 — Carry Signal (front-back proxy) + PPO
==========================================================================

策略简介：用收盘价与其长期均线作为front/back代理计算carry信号（反向基差），
         结合PPO确认动量方向。正carry表示价格高于长期水平（类似backwardation）。

使用指标：
  - Carry Signal: 用close vs SMA(60)作为front/back代理
  - PPO (fast=12, slow=26, signal=9): 百分比价格振荡器
  - ATR (period=14): trailing stop

进场条件：
  1. Carry Z-Score > 0.5（正carry，价格相对强势）
  2. PPO > 0（动量为正）

出场条件：
  1. ATR trailing stop (mult=4.0)
  2. PPO < -0.5（动量转弱）

优点：Carry信号 + 动量确认，多维度验证
缺点：单品种carry代理不如真实期限结构精确
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.spread.carry import carry_signal
from indicators.momentum.ppo import ppo
from indicators.trend.sma import sma
from indicators.volatility.atr import atr


class StrongTrendV183(TimeSeriesStrategy):
    """Carry信号(代理) + PPO动量策略。"""
    name = "strong_trend_v183"
    warmup = 60
    freq = "daily"

    sma_back_period: int = 60
    ppo_fast: int = 12
    ppo_slow: int = 26
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._carry_zscore = None
        self._ppo = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        # Use SMA as "back month" proxy
        back_proxy = sma(closes, period=self.sma_back_period)
        _, self._carry_zscore, _ = carry_signal(closes, back_proxy, period=20)
        self._ppo, _, _ = ppo(closes, fast=self.ppo_fast, slow=self.ppo_slow, signal=9)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        cz = self._carry_zscore[i]
        ppo_val = self._ppo[i]
        atr_val = self._atr[i]
        if np.isnan(cz) or np.isnan(ppo_val) or np.isnan(atr_val):
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
        if side == 0 and cz > 0.5 and ppo_val > 0.0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and ppo_val < -0.5:
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
