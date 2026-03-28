"""
QBase Strong Trend Strategy v180 — Volatility Seasonality + OI Momentum
=========================================================================

策略简介：当波动率处于季节性低位（预期将扩张）且持仓量动量走高时做多，
         捕捉波动率扩张初期的趋势启动。

使用指标：
  - Volatility Seasonality (vol_period=20): 波动率季节性评分
  - OI Momentum (period=20): 持仓量动量
  - ATR (period=14): trailing stop

进场条件：
  1. vol_seasonal_score < -0.5（波动率低于季节性常态，预期扩张）
  2. OI Momentum > 0（持仓量增长，资金流入）
  3. 价格 > 20日SMA（基本方向过滤）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. OI Momentum < -1.0（持仓量快速流出）

优点：波动率季节性 + OI流入，捕捉趋势启动
缺点：波动率季节性模式需要充足历史数据
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.seasonality.volatility_seasonality import vol_seasonality
from indicators.volume.oi_momentum import oi_momentum
from indicators.trend.sma import sma
from indicators.volatility.atr import atr


class StrongTrendV180(TimeSeriesStrategy):
    """波动率季节性 + OI动量策略。"""
    name = "strong_trend_v180"
    warmup = 60
    freq = "daily"

    oi_mom_period: int = 20
    sma_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._vol_score = None
        self._oi_mom = None
        self._sma = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        datetimes = context.get_full_datetime_array()
        oi = context.get_full_oi_array()

        self._vol_score, _ = vol_seasonality(closes, datetimes, vol_period=20)
        self._oi_mom = oi_momentum(oi, period=self.oi_mom_period)
        self._sma = sma(closes, period=self.sma_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        vs = self._vol_score[i]
        om = self._oi_mom[i]
        sma_val = self._sma[i]
        atr_val = self._atr[i]
        if np.isnan(vs) or np.isnan(om) or np.isnan(sma_val) or np.isnan(atr_val):
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
        if side == 0 and vs < -0.5 and om > 0.0 and price > sma_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and om < -1.0:
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
