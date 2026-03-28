"""Strong Trend v69 — Chaikin Oscillator + Fractal Levels."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.chaikin_oscillator import chaikin_oscillator
from indicators.trend.fractal import fractal_levels
from indicators.volatility.atr import atr


class StrongTrendV69(TimeSeriesStrategy):
    """
    策略简介：Chaikin Oscillator 资金流加速度 + Fractal Levels 结构突破策略。

    使用指标：
    - Chaikin Oscillator(3, 10): AD线的MACD，>0 资金加速流入
    - Fractal Levels: 分形高低点，价格突破分形高点 = 结构性突破
    - ATR(14): 追踪止损

    进场条件（做多）：
    - Chaikin Oscillator > 0 且上升（资金流加速）
    - 价格突破最近分形高点（结构性突破确认）

    出场条件：
    - ATR 追踪止损
    - Chaikin Oscillator 持续 < 0（资金流出加速）

    优点：分形突破 + 资金流确认，双重结构性验证
    缺点：分形高点在横盘时频繁更新，可能产生过多信号
    """
    name = "strong_trend_v69"
    warmup = 60
    freq = "daily"

    co_fast: int = 3
    co_slow: int = 10
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._co = None
        self._frac_high = None
        self._frac_low = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._co = chaikin_oscillator(highs, lows, closes, volumes,
                                       fast=self.co_fast, slow=self.co_slow)
        frac_h, frac_l = fractal_levels(highs, lows)
        self._frac_high = frac_h
        self._frac_low = frac_l
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        co_val = self._co[i]
        fh = self._frac_high[i]
        atr_val = self._atr[i]
        if np.isnan(co_val) or np.isnan(fh) or np.isnan(atr_val) or i < 2:
            return

        co_prev = self._co[i - 1]
        co_rising = co_val > co_prev

        # Stop loss
        if side == 1:
            self.highest = max(self.highest, price)
            trailing = self.highest - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Entry: Chaikin accelerating + fractal breakout
        if side == 0 and co_val > 0 and co_rising and price > fh:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: sustained money outflow
        elif side == 1 and co_val < 0 and not co_rising:
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
        return max(1, min(risk_lots, int(context.equity * 0.30 / margin)))

    def _reset(self):
        self.entry_price = 0.0
        self.highest = 0.0
        self.stop_price = 0.0
