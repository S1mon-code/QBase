"""
Boss Strategy v5 — VWAP Trend + Donchian
=========================================
Institutional money shows up in VWAP. Combine with channel for timing.
LONG ONLY. Supports scale-in (0-3).

Usage:
    ./run.sh strategies/boss/v5.py --symbols AG --freq 1h --start 2022
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
from indicators.volume.vwap import vwap
from indicators.volume.volume_momentum import volume_momentum

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BossV5(TimeSeriesStrategy):
    """
    策略简介：VWAP作为机构成本线确认趋势方向，Donchian突破精确入场，量能确认。
    交易哲学：VWAP是机构资金的"心理锚"。价格持续在VWAP上方意味着多头主导。
              配合Donchian突破，在机构支撑下的趋势中精确入场。
    使用指标：
      - VWAP: 成交量加权均价，机构成本参考线
      - Donchian(20): 通道突破入场信号
      - Volume Momentum(14): 量能确认，>1.2表示放量
      - ATR(14): 止损距离
    进场条件（做多）：
      1. 价格 > VWAP（在机构成本之上）
      2. 价格突破 Donchian(20) 上轨
      3. Volume Momentum > vol_mom_thresh（成交量放大确认）
    出场条件：
      - 价格连续 vwap_exit_bars 根bar收在VWAP下方
      - ATR追踪止损
      - 分层止盈（3ATR / 5ATR）
    优点：VWAP反映真实供需；Donchian突破过滤震荡；量能确认减少假突破
    缺点：VWAP从开盘累积，尾盘可能失真；1h频率交易成本较高
    """
    name = "boss_v5"
    warmup = 500
    freq = "1h"

    # Tunable parameters (<=5)
    don_period: int = 20
    vwap_exit_bars: int = 3
    vol_mom_thresh: float = 1.2
    atr_stop_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._vwap = None
        self._don_upper = None
        self._vol_mom = None
        self._atr = None
        self._avg_vol = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.bars_below_vwap = 0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._vwap = vwap(highs, lows, closes, volumes)
        don_upper, _, _ = donchian(highs, lows, self.don_period)
        self._don_upper = don_upper
        self._vol_mom = volume_momentum(volumes, period=14)
        self._atr = atr(highs, lows, closes, period=14)

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

        vwap_val = self._vwap[i]
        don_up = self._don_upper[i]
        vm = self._vol_mom[i]
        if np.isnan(vwap_val) or np.isnan(don_up) or np.isnan(vm):
            return

        self.bars_since_last_scale += 1

        # Track consecutive bars below VWAP
        if price < vwap_val:
            self.bars_below_vwap += 1
        else:
            self.bars_below_vwap = 0

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

        # 3. SIGNAL EXIT: consecutive bars below VWAP
        if side == 1:
            if self.bars_below_vwap >= self.vwap_exit_bars:
                context.close_long()
                self._reset_state()
                return

        # 4. ENTRY: price > VWAP + Donchian breakout + volume momentum
        if side == 0:
            if price <= vwap_val:
                return
            if price <= don_up:
                return
            if vm <= self.vol_mom_thresh:
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
        # Strategy-specific: still above VWAP + new Donchian high
        if price <= self._vwap[i]:
            return False
        if price <= self._don_upper[i]:
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
        self.bars_below_vwap = 0
