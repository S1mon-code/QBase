"""
QBase Strong Trend Strategy v195 — Connors RSI + OBV
=======================================================

策略简介：Connors RSI是复合RSI指标（结合短期RSI、连续涨跌天数RSI、百分位排名），
         结合OBV确认量价一致性。低Connors RSI + OBV上升 = 超卖反弹入场。

使用指标：
  - Connors RSI (rsi=3, streak=2, pct_rank=100): 复合RSI
  - OBV: 成交量平衡线
  - ATR (period=14): trailing stop

进场条件：
  1. Connors RSI从 < 20 上穿20（超卖反弹）
  2. OBV > OBV的20日SMA（量能支持上涨）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. Connors RSI > 80（超买区域）

优点：Connors RSI超卖信号精确，OBV确认
缺点：均值回归逻辑，强趋势中可能过早入场
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.momentum.connors_rsi import connors_rsi
from indicators.volume.obv import obv
from indicators.trend.sma import sma
from indicators.volatility.atr import atr


class StrongTrendV195(TimeSeriesStrategy):
    """Connors RSI超卖反弹 + OBV量能策略。"""
    name = "strong_trend_v195"
    warmup = 60
    freq = "daily"

    crsi_rsi: int = 3
    obv_sma_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._crsi = None
        self._obv = None
        self._obv_sma = None
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

        self._crsi = connors_rsi(closes, rsi=self.crsi_rsi, streak=2, pct_rank=100)
        self._obv = obv(closes, volumes)
        self._obv_sma = sma(self._obv, period=self.obv_sma_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if i < 1:
            return

        crsi_val = self._crsi[i]
        crsi_prev = self._crsi[i - 1]
        obv_val = self._obv[i]
        obv_s = self._obv_sma[i]
        atr_val = self._atr[i]
        if np.isnan(crsi_val) or np.isnan(crsi_prev) or np.isnan(obv_val) or np.isnan(obv_s) or np.isnan(atr_val):
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
        if side == 0 and crsi_val > 20 and crsi_prev <= 20 and obv_val > obv_s:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and crsi_val > 80:
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
