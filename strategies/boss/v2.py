"""
Boss Strategy v2 — Donchian + Volume Surge
===========================================
Shorter Donchian breakout with volume surge + ATR expansion for more frequent signals.
LONG ONLY. Supports scale-in (0-3).

Usage:
    ./run.sh strategies/boss/v2.py --symbols AG --freq 4h --start 2022
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
from indicators.trend.sma import sma

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BossV2(TimeSeriesStrategy):
    """
    策略简介：短周期Donchian突破+成交量激增+ATR扩张三重确认，4h频率产生更多交易信号。
    交易哲学：经典海龟突破的激进改良版——缩短通道周期(15)提高信号频率，
              用成交量激增过滤假突破，用ATR扩张确认波动率支持趋势。
              4h替代daily大幅增加bar数量和交易机会。
    使用指标：
      - Donchian(15): 入场通道（上轨突破买入，周期更短=更多信号）
      - Donchian(10): 出场通道（下轨跌破卖出）
      - Volume Spike(20, 1.5): 成交量激增检测
      - ATR(14) + SMA(ATR, 20): ATR扩张确认（ATR > 其20周期均线）
    进场条件（做多）：
      1. Close > Donchian(15) 上轨（价格突破）
      2. 最近3根bar内有成交量激增（volume spike）
      3. ATR[i] > SMA(ATR, 20)[i]（ATR在扩张，波动率支持趋势）
    出场条件：
      - Close < Donchian(10) 下轨（更快出场通道）
      - ATR追踪止损
      - 分层止盈（3ATR / 5ATR）
    优点：短通道+4h频率产生大量交易机会；三重过滤保持信号质量
    缺点：短通道假突破率较高；4h频率噪音多于daily
    """
    name = "boss_v2"
    warmup = 200
    freq = "4h"

    # Tunable parameters (<=5)
    don_entry: int = 15
    don_exit: int = 10
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._don_entry_upper = None
        self._don_exit_lower = None
        self._atr = None
        self._atr_sma = None
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
        # SMA of ATR for expansion detection
        self._atr_sma = sma(self._atr, period=20)
        # Lower threshold for volume spike (1.5x instead of 2.0x) = more signals
        self._vol_spikes = volume_spike(volumes, period=20, threshold=1.5)

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
        atr_sma_val = self._atr_sma[i]
        if np.isnan(don_upper) or np.isnan(don_exit_low) or np.isnan(atr_sma_val):
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

        # 4. ENTRY: Donchian breakout + volume surge + ATR expanding
        if side == 0:
            # Price above Donchian upper
            if price <= don_upper:
                return
            # ATR expanding: current ATR > its 20-period SMA
            if atr_val <= atr_sma_val:
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
        # Strategy-specific: still above Donchian upper + ATR still expanding
        don_upper = self._don_entry_upper[i]
        if np.isnan(don_upper) or price <= don_upper:
            return False
        atr_sma_val = self._atr_sma[i]
        if not np.isnan(atr_sma_val) and self._atr[i] <= atr_sma_val:
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
