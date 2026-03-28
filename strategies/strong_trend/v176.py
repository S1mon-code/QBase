"""
QBase Strong Trend Strategy v176 — Seasonal Momentum + Volume Momentum
=======================================================================

策略简介：结合季节性动量与成交量动量，当历史同期收益为正且成交量放大时做多，
         捕捉季节性规律与资金流入的共振信号。

使用指标：
  - Seasonal Momentum (lookback_years=3): 历史同期平均收益
  - Volume Momentum (period=14): 成交量动量确认
  - ATR (period=14): trailing stop

进场条件：
  1. 季节性20日预期收益 > 0（历史同期看涨）
  2. Volume Momentum > 1.2（放量确认）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. 季节性20日预期收益 < -0.005（历史同期看跌）

优点：季节性规律 + 量价共振，信号稳定
缺点：依赖历史季节性模式延续
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.seasonality.seasonal_momentum import seasonal_momentum
from indicators.volume.volume_momentum import volume_momentum
from indicators.volatility.atr import atr


class StrongTrendV176(TimeSeriesStrategy):
    """季节性动量 + 成交量动量共振策略。"""
    name = "strong_trend_v176"
    warmup = 60
    freq = "daily"

    lookback_years: int = 3
    vol_mom_period: int = 14
    vol_mom_thresh: float = 1.2
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._seasonal_5d = None
        self._seasonal_20d = None
        self._vol_mom = None
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
        datetimes = context.get_full_datetime_array()

        self._seasonal_5d, self._seasonal_20d = seasonal_momentum(
            closes, datetimes, lookback_years=self.lookback_years
        )
        self._vol_mom = volume_momentum(volumes, period=self.vol_mom_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        s20 = self._seasonal_20d[i]
        vm = self._vol_mom[i]
        atr_val = self._atr[i]
        if np.isnan(s20) or np.isnan(vm) or np.isnan(atr_val):
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
        if side == 0 and s20 > 0.0 and vm > self.vol_mom_thresh:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and s20 < -0.005:
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
