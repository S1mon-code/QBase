"""
Strong Trend Strategy v123 — Bollinger %B + OI Momentum
=========================================================
Uses Bollinger %B (price position within bands) to detect breakouts
above upper band, confirmed by OI momentum for new money entering.

  1. Bollinger %B   — price position relative to bands (>1 = above upper)
  2. OI Momentum    — open interest momentum confirms new positions

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v123.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.bollinger import bollinger_bands
from indicators.volume.oi_momentum import oi_momentum
from indicators.volatility.atr import atr


class StrongTrendV123(TimeSeriesStrategy):
    """
    策略简介：Bollinger %B突破 + OI动量确认的突破策略
    使用指标：Bollinger %B（价格在布林带中的位置）、OI Momentum（持仓动量）
    进场条件：%B > 1 (突破上轨) + OI动量 > 0 (新资金进场)
    出场条件：ATR trailing stop 或 %B < 0.5 (回到中轨以下)
    优点：%B > 1是强势突破信号，OI确认真实资金推动
    缺点：%B > 1可能是短暂的假突破，OI数据可能有延迟
    """
    name = "strong_trend_v123"
    warmup = 60
    freq = "daily"

    bb_period: int = 20
    bb_std: float = 2.0
    oi_period: int = 20
    pctb_entry: float = 1.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._bb_upper = None
        self._bb_mid = None
        self._bb_lower = None
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

        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            closes, period=self.bb_period, num_std=self.bb_std
        )
        self._oi_mom = oi_momentum(oi, self.oi_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        bb_upper = self._bb_upper[i]
        bb_lower = self._bb_lower[i]
        bb_mid = self._bb_mid[i]
        oi_val = self._oi_mom[i]
        atr_val = self._atr[i]

        if np.isnan(bb_upper) or np.isnan(bb_lower) or np.isnan(atr_val):
            return

        # Compute %B: (price - lower) / (upper - lower)
        bb_range = bb_upper - bb_lower
        if bb_range <= 0:
            return
        pct_b = (price - bb_lower) / bb_range

        oi_positive = not np.isnan(oi_val) and oi_val > 0

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
        if side == 0 and pct_b > self.pctb_entry and oi_positive:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and pct_b < 0.5:
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
