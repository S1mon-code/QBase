"""
Boss Strategy v9 — KAMA Adaptive + BB Breakout
================================================
KAMA adapts to market conditions. Only enter when KAMA confirms AND
Bollinger breakout triggers.
LONG ONLY. Supports scale-in (0-3).

Usage:
    ./run.sh strategies/boss/v9.py --symbols AG --freq 4h --start 2022
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
from indicators.volatility.bollinger import bollinger_bands
from indicators.momentum.kama import kama
from indicators.volume.oi_momentum import oi_momentum

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BossV9(TimeSeriesStrategy):
    """
    策略简介：KAMA自适应均线确认趋势，布林带突破触发入场，双重自适应过滤。
    交易哲学：市场噪音大时KAMA变慢（过滤噪音），趋势明确时KAMA变快（紧跟趋势）。
              当这样一个"智能"均线上升+价格突破布林上轨，这是高确信度的趋势信号。
              OI Momentum确认新资金进场而非存量博弈。
    使用指标：
      - KAMA(20): Kaufman自适应均线，根据效率比自动调整灵敏度
      - Bollinger Bands(20, 2.0): 波动率通道，上轨突破=动能释放
      - OI Momentum(20): 持仓量变化率，>0表示新资金入场
      - ATR(14): 止损距离 + KAMA平坦判断
    进场条件（做多）：
      1. KAMA 上升中（KAMA[i] > KAMA[i-3]）
      2. 价格 > 布林上轨（波动率突破）
      3. OI Momentum > 0（新资金入场确认）
    出场条件：
      - KAMA 平坦化（|KAMA[i] - KAMA[i-3]| < kama_flat_mult * ATR）
      - ATR追踪止损
      - 分层止盈（3ATR / 5ATR）
    优点：KAMA的自适应特性天然过滤震荡；OI确认新资金=趋势有后续动力
    缺点：KAMA在趋势转折点反应仍有延迟；OI数据可能不是所有品种都可靠
    """
    name = "boss_v9"
    warmup = 200
    freq = "4h"

    # Tunable parameters (<=5)
    kama_period: int = 20
    bb_period: int = 20
    kama_flat_mult: float = 0.1
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._kama = None
        self._bb_upper = None
        self._oi_mom = None
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

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()

        self._kama = kama(closes, period=self.kama_period)
        bb_upper, _, _ = bollinger_bands(closes, self.bb_period, 2.0)
        self._bb_upper = bb_upper
        self._oi_mom = oi_momentum(oi, period=20)
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
        if i < 3:
            return

        kama_now = self._kama[i]
        kama_prev3 = self._kama[i - 3]
        bb_up = self._bb_upper[i]
        oi_mom_val = self._oi_mom[i]
        if np.isnan(kama_now) or np.isnan(kama_prev3) or np.isnan(bb_up):
            return

        # OI momentum might be NaN if no OI data — treat as neutral (allow entry)
        oi_ok = True
        if not np.isnan(oi_mom_val):
            oi_ok = oi_mom_val > 0

        self.bars_since_last_scale += 1

        kama_rising = kama_now > kama_prev3
        kama_change = abs(kama_now - kama_prev3)
        kama_flat = kama_change < self.kama_flat_mult * atr_val

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

        # 3. SIGNAL EXIT: KAMA flattens → trend losing momentum
        if side == 1:
            if kama_flat:
                context.close_long()
                self._reset_state()
                return

        # 4. ENTRY: KAMA rising + price > BB upper + OI momentum > 0
        if side == 0:
            if not kama_rising:
                return
            if price <= bb_up:
                return
            if not oi_ok:
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
        # Strategy-specific: KAMA still rising + price still above BB upper
        if i < 3:
            return False
        if self._kama[i] <= self._kama[i - 3]:
            return False
        if price <= self._bb_upper[i]:
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
