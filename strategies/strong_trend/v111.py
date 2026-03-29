"""
Strong Trend Strategy v111 — Rogers-Satchell Vol Expansion + PPO
==================================================================
Uses drift-independent Rogers-Satchell volatility for expansion detection,
confirmed by PPO (Percentage Price Oscillator) for momentum direction.

  1. Rogers-Satchell — drift-independent OHLC volatility estimator
  2. PPO             — percentage-based momentum oscillator

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v111.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.rogers_satchell import rogers_satchell
from indicators.momentum.ppo import ppo
from indicators.volatility.atr import atr


class StrongTrendV111(TimeSeriesStrategy):
    """
    策略简介：Rogers-Satchell波动率扩张 + PPO动量确认的突破策略
    使用指标：Rogers-Satchell Vol（漂移无关波动率）、PPO（百分比动量）
    进场条件：RS Vol扩张 + PPO > 0且上升 + PPO > Signal线
    出场条件：ATR trailing stop 或 PPO < 0
    优点：RS Vol对有漂移的市场无偏，PPO标准化便于比较
    缺点：RS Vol计算较复杂，PPO滞后
    """
    name = "strong_trend_v111"
    warmup = 60
    freq = "daily"

    rs_period: int = 20
    ppo_fast: int = 12
    ppo_slow: int = 26
    rs_expansion: float = 1.4
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._rs = None
        self._ppo = None
        self._ppo_signal = None
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

        self._rs = rogers_satchell(opens, highs, lows, closes, period=self.rs_period)
        self._ppo, self._ppo_signal, _ = ppo(closes, fast=self.ppo_fast, slow=self.ppo_slow)
        self._atr = atr(highs, lows, closes, period=14)

        # Rolling average of RS vol
        n = len(closes)
        self._rs_avg = np.full(n, np.nan)
        for k in range(39, n):
            window = self._rs[k - 39:k + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                self._rs_avg[k] = np.mean(valid)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        cur_rs = self._rs[i]
        rs_avg = self._rs_avg[i]
        cur_ppo = self._ppo[i]
        cur_ppo_sig = self._ppo_signal[i]
        atr_val = self._atr[i]

        if np.isnan(cur_rs) or np.isnan(rs_avg) or np.isnan(cur_ppo) or np.isnan(cur_ppo_sig) or np.isnan(atr_val):
            return

        rs_expanding = rs_avg > 0 and cur_rs > self.rs_expansion * rs_avg
        ppo_bullish = cur_ppo > 0 and cur_ppo > cur_ppo_sig

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
        if side == 0 and rs_expanding and ppo_bullish:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and cur_ppo < 0:
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
