"""
Strong Trend Strategy v115 — Conditional Volatility Shift + Force Index
=========================================================================
Detects asymmetric volatility shifts (upside vol increasing relative to
downside) combined with Force Index for volume-weighted momentum.

  1. Conditional Volatility — up_vol vs down_vol asymmetry detection
  2. Force Index           — volume-weighted price momentum

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v115.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.conditional_vol import conditional_volatility
from indicators.volume.force_index import force_index
from indicators.volatility.atr import atr


class StrongTrendV115(TimeSeriesStrategy):
    """
    策略简介：条件波动率非对称转变 + Force Index动量确认的突破策略
    使用指标：Conditional Volatility（上行/下行波动率分离）、Force Index（量价动量）
    进场条件：上行波动率 > 下行波动率（vol_asymmetry < 0）+ Force Index > 0
    出场条件：ATR trailing stop 或 Force Index < 0
    优点：区分上行和下行波动率，捕捉看涨波动率扩张
    缺点：条件波动率需要足够样本量，短期可能不稳定
    """
    name = "strong_trend_v115"
    warmup = 60
    freq = "daily"

    cv_period: int = 20
    fi_period: int = 13
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._up_vol = None
        self._down_vol = None
        self._vol_asym = None
        self._fi = None
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

        self._up_vol, self._down_vol, self._vol_asym = conditional_volatility(
            closes, period=self.cv_period
        )
        self._fi = force_index(closes, volumes, period=self.fi_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        vol_asym = self._vol_asym[i]
        up_vol = self._up_vol[i]
        fi_val = self._fi[i]
        atr_val = self._atr[i]

        if np.isnan(vol_asym) or np.isnan(fi_val) or np.isnan(atr_val) or np.isnan(up_vol):
            return

        # Upside vol > downside vol means vol_asymmetry < 0 (down_vol - up_vol)
        bullish_vol_shift = vol_asym < 0 and up_vol > 0
        force_positive = fi_val > 0

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
        if side == 0 and bullish_vol_shift and force_positive:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and fi_val < 0:
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
