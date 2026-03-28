"""
QBase Strong Trend Strategy v94 — Hurst R/S + OI Flow
=======================================================

策略简介：Hurst R/S 分析（Rescaled Range）检测长记忆性趋势，
         OI Flow 确认资金流入方向后入场。

使用指标：
  - Hurst R/S (min_period=10, max_period=100): H > 0.55 = 趋势
  - OI Flow (period=14): 资金流向确认
  - ATR (period=14): trailing stop

进场条件：
  1. Hurst R/S > 0.55（长记忆趋势）
  2. OI Flow > 0（资金流入）
  3. 收盘价 > 20日均线

出场条件：
  1. ATR trailing stop（mult=4.5）
  2. OI Flow < -1.0（资金大幅流出）

优点：R/S 分析比简单 Hurst 更稳健，OI 数据是中国期货独有优势
缺点：R/S 计算较慢，OI 数据可能有延迟
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.hurst_rs import hurst_rs
from indicators.volume.oi_flow import oi_flow
from indicators.volatility.atr import atr
from indicators.trend.sma import sma


class StrongTrendV94(TimeSeriesStrategy):
    """Hurst R/S trend detection + OI Flow confirmation."""
    name = "strong_trend_v94"
    warmup = 60
    freq = "daily"

    hurst_threshold: float = 0.55
    oi_flow_period: int = 14
    sma_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._hurst = None
        self._oi_flow = None
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
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()

        self._hurst = hurst_rs(closes, min_period=10, max_period=100)
        self._oi_flow = oi_flow(closes, oi, volumes, period=self.oi_flow_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._sma = sma(closes, self.sma_period)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        h_val = self._hurst[i]
        flow_val = self._oi_flow[i]
        atr_val = self._atr[i]
        sma_val = self._sma[i]
        if np.isnan(h_val) or np.isnan(flow_val) or np.isnan(atr_val) or np.isnan(sma_val):
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
        if side == 0 and h_val > self.hurst_threshold and flow_val > 0 and price > sma_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and flow_val < -1.0:
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
