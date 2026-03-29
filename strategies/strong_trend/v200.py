"""
QBase Strong Trend Strategy v200 — Ultimate Oscillator + OI Momentum + Supertrend (3-indicator)
=================================================================================================

策略简介：三指标终极策略。Ultimate Oscillator综合多周期买压/卖压，OI Momentum确认
         持仓量增长，Supertrend确认趋势方向。三重共振产生高置信度入场信号。

使用指标：
  - Ultimate Oscillator (7, 14, 28): 多周期振荡器
  - OI Momentum (period=20): 持仓量动量
  - Supertrend (period=10, mult=3.0): 趋势方向
  - ATR (period=14): trailing stop

进场条件：
  1. Ultimate Oscillator > 50（买压占优）
  2. OI Momentum > 0（持仓量增长）
  3. Supertrend方向为多头（dir == 1）

出场条件：
  1. ATR trailing stop (mult=4.0)
  2. Supertrend翻空（dir == -1）

优点：三指标三维度（动量+资金+趋势）全面确认
缺点：三重过滤可能导致信号过少
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.momentum.ultimate_oscillator import ultimate_oscillator
from indicators.volume.oi_momentum import oi_momentum
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr


class StrongTrendV200(TimeSeriesStrategy):
    """Ultimate Oscillator + OI Momentum + Supertrend三重共振策略。"""
    name = "strong_trend_v200"
    warmup = 60
    freq = "daily"

    oi_mom_period: int = 20
    st_period: int = 10
    st_mult: float = 3.0
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._uo = None
        self._oi_mom = None
        self._st_dir = None
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

        self._uo = ultimate_oscillator(highs, lows, closes, 7, 14, 28)
        self._oi_mom = oi_momentum(oi, period=self.oi_mom_period)
        _, self._st_dir = supertrend(highs, lows, closes,
                                      period=self.st_period, mult=self.st_mult)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        uo_val = self._uo[i]
        om = self._oi_mom[i]
        st_d = self._st_dir[i]
        atr_val = self._atr[i]
        if np.isnan(uo_val) or np.isnan(om) or np.isnan(st_d) or np.isnan(atr_val):
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
        if side == 0 and uo_val > 50.0 and om > 0.0 and st_d == 1:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and st_d == -1:
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
