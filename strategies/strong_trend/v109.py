"""
Strong Trend Strategy v109 — Garman-Klass Volatility Spike + Volume Spike
===========================================================================
Detects OHLC-based volatility spikes using Garman-Klass estimator,
confirmed by volume spikes indicating institutional breakout activity.

  1. Garman-Klass Vol — efficient OHLC volatility spike detection
  2. Volume Spike     — confirms conviction behind the breakout

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v109.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.garman_klass import garman_klass
from indicators.volume.volume_spike import volume_spike
from indicators.volatility.atr import atr


class StrongTrendV109(TimeSeriesStrategy):
    """
    策略简介：Garman-Klass波动率突增 + 成交量突增的突破策略
    使用指标：Garman-Klass Vol（高效波动率）、Volume Spike（量能突增）
    进场条件：GK Vol > 1.5倍均值 + 成交量突增 + 价格上涨
    出场条件：ATR trailing stop 或 GK Vol回落到均值以下
    优点：GK Vol比收盘价波动率效率高7.4倍，量价共振强
    缺点：GK假设零漂移，强趋势中可能低估波动率
    """
    name = "strong_trend_v109"
    warmup = 60
    freq = "daily"

    gk_period: int = 20
    vol_spike_threshold: float = 2.0
    gk_expansion: float = 1.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._gk = None
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
        opens = context.get_full_open_array()
        volumes = context.get_full_volume_array()

        self._gk = garman_klass(opens, highs, lows, closes, period=self.gk_period)
        self._vol_spike = volume_spike(volumes, period=20, threshold=self.vol_spike_threshold)
        self._atr = atr(highs, lows, closes, period=14)

        # Pre-compute GK rolling average
        n = len(closes)
        self._gk_avg = np.full(n, np.nan)
        for k in range(39, n):
            window = self._gk[k - 39:k + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                self._gk_avg[k] = np.mean(valid)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        if i < 1:
            return

        cur_gk = self._gk[i]
        gk_avg = self._gk_avg[i]
        atr_val = self._atr[i]

        if np.isnan(cur_gk) or np.isnan(gk_avg) or np.isnan(atr_val):
            return

        gk_spiking = gk_avg > 0 and cur_gk > self.gk_expansion * gk_avg
        # Volume spike in last 3 bars
        recent_vol_spike = np.any(self._vol_spike[max(0, i - 2):i + 1])
        # Price going up
        price_up = price > context.get_full_close_array()[i - 1]

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
        if side == 0 and gk_spiking and recent_vol_spike and price_up:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and gk_avg > 0 and cur_gk < 0.7 * gk_avg:
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
