"""
Boss Strategy v2 — Donchian Turtle Revival
===========================================
Classic turtle breakout with volatility expansion filter.
LONG ONLY. Supports scale-in (0-3).

Usage:
    ./run.sh strategies/boss/v2.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.trend.donchian import donchian
from indicators.volume.volume_spike import volume_spike

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BossV2(TimeSeriesStrategy):
    """
    策略简介：经典海龟突破策略的现代改良版，加入波动率扩张过滤和成交量确认。
    交易哲学：趋势的起点是突破。40年历史证明Donchian突破在强趋势品种上有效。
    使用指标：
      - Donchian(20): 入场通道（上轨突破买入）
      - Donchian(10): 出场通道（下轨跌破卖出，更快反应）
      - ATR(14): 波动率扩张过滤 + 止损距离
      - Volume Spike: 成交量放大确认突破真实性
    进场条件（做多）：
      1. 价格突破 Donchian(20) 上轨
      2. ATR 扩张中（当前ATR > 5根bar前ATR * atr_expand_mult）
      3. 最近3根bar内有成交量放大（volume spike）
    出场条件：
      - 价格跌破 Donchian(10) 下轨（更快的出场通道）
      - ATR追踪止损
      - 分层止盈（3ATR / 5ATR）
    优点：趋势起点入场，持仓时间长，利润空间大
    缺点：突破假信号多，震荡市连续止损；入场不够精确
    """
    name = "boss_v2"
    warmup = 120
    freq = "daily"

    # Tunable parameters (<=5)
    don_entry: int = 20
    don_exit: int = 10
    atr_expand_mult: float = 1.1
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._don_entry_upper = None
        self._don_entry_lower = None
        self._don_exit_lower = None
        self._atr = None
        self._vol_spikes = None
        self._avg_vol = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        entry_upper, entry_lower, _ = donchian(highs, lows, self.don_entry)
        self._don_entry_upper = entry_upper
        _, exit_lower, _ = donchian(highs, lows, self.don_exit)
        self._don_exit_lower = exit_lower
        self._atr = atr(highs, lows, closes, period=14)
        self._vol_spikes = volume_spike(volumes, period=20, threshold=2.0)

        # avg volume
        n = len(volumes)
        cumsum = np.cumsum(np.insert(volumes, 0, 0.0))
        self._avg_vol = np.full(n, np.nan)
        if n >= 20:
            self._avg_vol[20:] = (cumsum[21:n + 1] - cumsum[1:n - 19]) / 20

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_vol[i]) and context.volume < self._avg_vol[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        don_upper = self._don_entry_upper[i]
        don_exit_low = self._don_exit_lower[i]
        if np.isnan(don_upper) or np.isnan(don_exit_low):
            return

        self.bars_since_last_scale += 1

        # 1. STOP LOSS
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

        # 2. TIERED PROFIT-TAKING
        if side == 1 and self.entry_price > 0:
            profit_atr = (price - self.entry_price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_long(lots=max(1, lots // 3))
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_long(lots=max(1, lots // 3))
                self._took_profit_3atr = True
                return

        # 3. SIGNAL EXIT: price breaks below Donchian(10) lower
        if side == 1:
            if price < don_exit_low:
                context.close_long()
                self._reset_state()
                return

        # 4. ENTRY: Donchian breakout + ATR expanding + volume spike
        if side == 0:
            if price <= don_upper:
                return
            # ATR expanding
            if i < 5:
                return
            atr_prev = self._atr[i - 5]
            if np.isnan(atr_prev) or atr_val < atr_prev * self.atr_expand_mult:
                return
            # Volume spike in last 3 bars
            start_idx = max(0, i - 2)
            if not np.any(self._vol_spikes[start_idx:i + 1]):
                return

            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. SCALE-IN
        elif side == 1 and self._should_add(i, price, atr_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, i, price, atr_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        # Strategy-specific: new Donchian high and ATR still expanding
        don_upper = self._don_entry_upper[i]
        if np.isnan(don_upper) or price <= don_upper:
            return False
        if i >= 5:
            atr_prev = self._atr[i - 5]
            if not np.isnan(atr_prev) and self._atr[i] < atr_prev:
                return False
        return True

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_stop_mult * atr_val * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = context.close_raw * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        return max(1, min(risk_lots, int(context.equity * 0.30 / margin)))

    def _reset_state(self):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
