"""
Strong Trend Strategy v104 — Keltner Width Narrowing→Expansion + OI Momentum
==============================================================================
Monitors Keltner Channel width for compression-to-expansion transitions,
confirmed by open interest momentum indicating new money flow.

  1. Keltner Width   — bandwidth narrowing then expanding = breakout
  2. OI Momentum     — rising OI confirms new positions entering

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v104.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.keltner_width import keltner_width
from indicators.volume.oi_momentum import oi_momentum
from indicators.volatility.atr import atr


class StrongTrendV104(TimeSeriesStrategy):
    """
    策略简介：Keltner通道宽度收窄→扩张 + OI动量确认的突破策略
    使用指标：Keltner Width（通道宽度）、OI Momentum（持仓动量）
    进场条件：Width从低位扩张 + OI动量为正 + 价格上行
    出场条件：ATR trailing stop 或 OI动量转负
    优点：Keltner比Bollinger更平滑，OI确认真实资金流入
    缺点：需要OI数据，部分品种OI质量差
    """
    name = "strong_trend_v104"
    warmup = 60
    freq = "daily"

    kw_ema: int = 20
    kw_atr: int = 10
    oi_period: int = 20
    atr_trail_mult: float = 4.5
    width_expansion: float = 1.3

    def __init__(self):
        super().__init__()
        self._kw = None
        self._oi_mom = None
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

        self._kw = keltner_width(highs, lows, closes, ema_period=self.kw_ema, atr_period=self.kw_atr)
        self._oi_mom = oi_momentum(oi, self.oi_period)
        self._atr = atr(highs, lows, closes, period=14)

        # Pre-compute rolling min of Keltner width (20-bar)
        n = len(closes)
        self._kw_min20 = np.full(n, np.nan)
        for k in range(19, n):
            window = self._kw[k - 19:k + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                self._kw_min20[k] = np.min(valid)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        cur_kw = self._kw[i]
        oi_val = self._oi_mom[i]
        atr_val = self._atr[i]
        kw_min = self._kw_min20[i]

        if np.isnan(cur_kw) or np.isnan(oi_val) or np.isnan(atr_val) or np.isnan(kw_min):
            return

        # Width expanding from recent low
        width_expanding = kw_min > 0 and cur_kw > self.width_expansion * kw_min

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
        if side == 0 and width_expanding and oi_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and oi_val < -1.0:
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
