"""
QBase Strong Trend Strategy v184 — Term Premium Proxy + Volume Spike
======================================================================

策略简介：用收盘价与长期EMA的差作为期限溢价代理，结合成交量突增确认资金涌入。
         正溢价(价格>长期均线)表示市场看涨预期。

使用指标：
  - Term Premium: close vs EMA(60)作为front/back代理
  - Volume Spike (period=20, threshold=2.0): 成交量突增检测
  - ATR (period=14): trailing stop

进场条件：
  1. Term Premium Z-Score > 0.5（价格高于长期水平）
  2. Volume Spike == 1（成交量突增）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. Term Premium Z-Score < -1.0（价格大幅低于长期水平）

优点：结构性溢价 + 量能突破
缺点：Volume Spike信号稀疏
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.spread.term_premium import term_premium
from indicators.volume.volume_spike import volume_spike
from indicators.trend.ema import ema
from indicators.volatility.atr import atr


class StrongTrendV184(TimeSeriesStrategy):
    """期限溢价代理 + 成交量突增策略。"""
    name = "strong_trend_v184"
    warmup = 60
    freq = "daily"

    ema_back_period: int = 60
    vol_spike_thresh: float = 2.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._tp_zscore = None
        self._vol_spike = None
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

        back_proxy = ema(closes, period=self.ema_back_period)
        _, self._tp_zscore, _ = term_premium(closes, back_proxy, period=20)
        self._vol_spike = volume_spike(volumes, period=20, threshold=self.vol_spike_thresh)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        tp_z = self._tp_zscore[i]
        vs = self._vol_spike[i]
        atr_val = self._atr[i]
        if np.isnan(tp_z) or np.isnan(vs) or np.isnan(atr_val):
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
        if side == 0 and tp_z > 0.5 and vs == 1:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and tp_z < -1.0:
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
