"""
Boss Strategy v6 — HMA Regime Shift
====================================
HMA is the fastest responsive MA. Use it to detect regime shifts early.
LONG ONLY. Supports scale-in (0-3).

Usage:
    ./run.sh strategies/boss/v6.py --symbols AG --freq 4h --start 2022
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
from indicators.trend.hma import hma
from indicators.volume.force_index import force_index

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BossV6(TimeSeriesStrategy):
    """
    策略简介：利用HMA（Hull均线）的超低滞后性，早期捕捉趋势政权转变，Force Index确认多头力量。
    交易哲学：传统均线滞后严重，趋势走了一半才发出信号。HMA用加权差分消除滞后，
              能在趋势诞生初期就检测到方向变化。配合ATR扩张和Force Index，三重确认。
    使用指标：
      - HMA(20): Hull均线，低滞后趋势方向，斜率判断政权
      - ATR(14): 波动率扩张确认 + 止损距离
      - Force Index(13): 力量指数，>0表示多头主导
    进场条件（做多）：
      1. HMA斜率连续3根bar为正（HMA[i] > HMA[i-1] > HMA[i-2]）
      2. ATR扩张中（ATR[i] > ATR[i-5]）
      3. Force Index > 0（多头力量确认）
    出场条件：
      - HMA斜率连续2根bar为负（HMA[i] < HMA[i-1] < HMA[i-2]）
      - ATR追踪止损
      - 分层止盈（3ATR / 5ATR）
    优点：HMA反应极快，趋势初期入场；三重确认降低假信号
    缺点：HMA在窄幅震荡中可能频繁翻转；Force Index受大单影响波动大
    """
    name = "boss_v6"
    warmup = 200
    freq = "4h"

    # Tunable parameters (<=5)
    hma_period: int = 20
    fi_period: int = 13
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._hma = None
        self._fi = None
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

        self._hma = hma(closes, self.hma_period)
        self._fi = force_index(closes, volumes, period=self.fi_period)
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
        if i < 5:
            return

        hma_now = self._hma[i]
        hma_1 = self._hma[i - 1]
        hma_2 = self._hma[i - 2]
        fi_val = self._fi[i]
        if np.isnan(hma_now) or np.isnan(hma_1) or np.isnan(hma_2) or np.isnan(fi_val):
            return

        self.bars_since_last_scale += 1

        # HMA slope analysis
        hma_rising_3 = (hma_now > hma_1 > hma_2)
        hma_falling_2 = (hma_now < hma_1 < hma_2)

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

        # 3. SIGNAL EXIT: HMA slope negative for 2 consecutive bars
        if side == 1:
            if hma_falling_2:
                context.close_long()
                self._reset_state()
                return

        # 4. ENTRY: HMA rising 3 bars + ATR expanding + Force Index > 0
        if side == 0:
            if not hma_rising_3:
                return
            # ATR expanding
            atr_prev = self._atr[i - 5]
            if np.isnan(atr_prev) or atr_val <= atr_prev:
                return
            # Force Index positive
            if fi_val <= 0:
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
        # Strategy-specific: HMA still rising + Force Index still positive
        if i < 1:
            return False
        if self._hma[i] <= self._hma[i - 1]:
            return False
        if self._fi[i] <= 0:
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
