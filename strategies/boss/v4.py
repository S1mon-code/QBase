"""
Boss Strategy v4 — Keltner + EMA Cloud
=======================================
Dual confirmation: EMA cloud for trend + Keltner breakout for timing.
LONG ONLY. Supports scale-in (0-3).

Usage:
    ./run.sh strategies/boss/v4.py --symbols AG --freq 4h --start 2022
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
from indicators.trend.ema import ema
from indicators.trend.keltner import keltner
from indicators.volume.mfi import mfi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BossV4(TimeSeriesStrategy):
    """
    策略简介：EMA云层确认趋势方向，Keltner通道突破精确择时，MFI确认买方压力。
    交易哲学：双重确认降低假信号。趋势方向由均线定义，入场时机由波动率通道决定。
    使用指标：
      - EMA(20) / EMA(50): 均线云层，快线在慢线上方=趋势看多
      - Keltner Channel(20, 2.0): 波动率通道，上轨突破=动能释放
      - MFI(14): 资金流量指数，>50表示买方主导
      - ATR(14): 止损距离
    进场条件（做多）：
      1. EMA(20) > EMA(50)（均线云看多）
      2. 价格突破 Keltner 上轨（波动率突破）
      3. MFI > 50（买方压力）
    出场条件：
      - 价格跌回 Keltner 中轨以下 OR EMA(20) < EMA(50)
      - ATR追踪止损
      - 分层止盈（3ATR / 5ATR）
    优点：双重确认体系大幅降低假突破率；Keltner基于ATR自适应波动率
    缺点：多重过滤可能导致入场偏晚；均线交叉滞后性
    """
    name = "boss_v4"
    warmup = 200
    freq = "4h"

    # Tunable parameters (<=5)
    kc_period: int = 20
    kc_mult: float = 2.0
    ema_fast: int = 20
    ema_slow: int = 50
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._ema_fast = None
        self._ema_slow = None
        self._kc_upper = None
        self._kc_middle = None
        self._mfi = None
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

        self._ema_fast = ema(closes, self.ema_fast)
        self._ema_slow = ema(closes, self.ema_slow)
        kc_upper, kc_middle, kc_lower = keltner(
            highs, lows, closes,
            ema_period=self.kc_period, atr_period=self.kc_period,
            multiplier=self.kc_mult
        )
        self._kc_upper = kc_upper
        self._kc_middle = kc_middle
        self._mfi = mfi(highs, lows, closes, volumes, period=14)
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

        ema_f = self._ema_fast[i]
        ema_s = self._ema_slow[i]
        kc_up = self._kc_upper[i]
        kc_mid = self._kc_middle[i]
        mfi_val = self._mfi[i]
        if np.isnan(ema_f) or np.isnan(ema_s) or np.isnan(kc_up) or np.isnan(mfi_val):
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

        # 3. SIGNAL EXIT: price below Keltner middle OR EMA cross bearish
        if side == 1:
            if price < kc_mid or ema_f < ema_s:
                context.close_long()
                self._reset_state()
                return

        # 4. ENTRY: EMA cloud bullish + Keltner upper breakout + MFI > 50
        if side == 0:
            if ema_f <= ema_s:
                return
            if price <= kc_up:
                return
            if mfi_val <= 50:
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
        # Strategy-specific: still above Keltner upper + EMA cloud bullish + MFI bullish
        if price <= self._kc_upper[i]:
            return False
        if self._ema_fast[i] <= self._ema_slow[i]:
            return False
        if self._mfi[i] <= 50:
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
