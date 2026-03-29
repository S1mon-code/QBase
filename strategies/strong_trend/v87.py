"""
QBase Strong Trend Strategy v87 — Yang-Zhang Volatility Expansion + OBV
=========================================================================

策略简介：Yang-Zhang 波动率扩张检测趋势启动 regime，
         OBV 确认量能方向后入场。

使用指标：
  - Yang-Zhang Volatility (period=20): 波动率扩张 = 趋势启动
  - OBV: 量价方向确认
  - ATR (period=14): trailing stop

进场条件：
  1. YZ Vol 当前值 > 1.5 * YZ Vol 20日均值（波动率扩张）
  2. OBV 上升（OBV > OBV[-5]）
  3. 收盘价 > 20日均线

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. OBV 下降超过 5 日（量能衰退）

优点：Yang-Zhang 波动率比传统 close-to-close 更精确，OBV 简单有效
缺点：波动率扩张也可能是下跌趋势
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.yang_zhang import yang_zhang
from indicators.volume.obv import obv
from indicators.volatility.atr import atr
from indicators.trend.sma import sma


class StrongTrendV87(TimeSeriesStrategy):
    """Yang-Zhang volatility expansion + OBV direction."""
    name = "strong_trend_v87"
    warmup = 60
    freq = "daily"

    yz_period: int = 20
    vol_expand_mult: float = 1.5
    obv_lookback: int = 5
    sma_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._yz = None
        self._obv = None
        self._atr = None
        self._sma = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        opens = context.get_full_open_array()
        volumes = context.get_full_volume_array()

        self._yz = yang_zhang(opens, highs, lows, closes, period=self.yz_period)
        self._obv = obv(closes, volumes)
        self._atr = atr(highs, lows, closes, period=14)
        self._sma = sma(closes, self.sma_period)
        self._yz_sma = sma(self._yz, self.yz_period)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        yz_val = self._yz[i]
        yz_avg = self._yz_sma[i]
        obv_val = self._obv[i]
        atr_val = self._atr[i]
        sma_val = self._sma[i]
        if np.isnan(yz_val) or np.isnan(yz_avg) or np.isnan(obv_val) or np.isnan(atr_val) or np.isnan(sma_val):
            return
        if i < self.obv_lookback:
            return

        obv_prev = self._obv[i - self.obv_lookback]
        if np.isnan(obv_prev):
            return

        vol_expanding = yz_val > self.vol_expand_mult * yz_avg
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
        if side == 0 and vol_expanding and obv_rising and price > sma_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and obv_val < obv_prev:
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
