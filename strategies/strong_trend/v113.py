"""
Strong Trend Strategy v113 — Chaikin Volatility + OBV
======================================================
Uses Chaikin Volatility (rate of change of high-low EMA spread) to
detect volatility expansion, with OBV confirming accumulation.

  1. Chaikin Volatility — ROC of EMA(H-L), expansion = widening ranges
  2. OBV               — on-balance volume for accumulation confirmation

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v113.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.chaikin_vol import chaikin_volatility
from indicators.volume.obv import obv
from indicators.volatility.atr import atr


class StrongTrendV113(TimeSeriesStrategy):
    """
    策略简介：Chaikin波动率扩张 + OBV积累确认的突破策略
    使用指标：Chaikin Volatility（价差变化率）、OBV（能量潮）
    进场条件：Chaikin Vol > 阈值 + OBV上升 + 价格上行
    出场条件：ATR trailing stop 或 OBV持续下降
    优点：Chaikin Vol直接衡量波幅扩张速度，OBV简单有效
    缺点：Chaikin Vol可能在下跌中也为正
    """
    name = "strong_trend_v113"
    warmup = 60
    freq = "daily"

    cv_ema: int = 10
    cv_roc: int = 10
    obv_lookback: int = 10
    cv_threshold: float = 20.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._cv = None
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

        self._cv = chaikin_volatility(highs, lows, ema=self.cv_ema, roc=self.cv_roc)
        self._obv = obv(closes, volumes)
        self._atr = atr(highs, lows, closes, period=14)
        self._closes = closes

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        if i < self.obv_lookback:
            return

        cv_val = self._cv[i]
        atr_val = self._atr[i]

        if np.isnan(cv_val) or np.isnan(atr_val):
            return

        obv_rising = (
            not np.isnan(self._obv[i]) and not np.isnan(self._obv[i - self.obv_lookback])
            and self._obv[i] > self._obv[i - self.obv_lookback]
        )
        price_up = price > self._closes[i - 1]

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
        if side == 0 and cv_val > self.cv_threshold and obv_rising and price_up:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1:
            obv_falling = (
                not np.isnan(self._obv[i]) and not np.isnan(self._obv[i - self.obv_lookback])
                and self._obv[i] < self._obv[i - self.obv_lookback]
            )
            if obv_falling and cv_val < 0:
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
