"""Strong Trend v147 — High-Low Spread + ALMA trend filter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.microstructure.high_low_spread import hl_spread
from indicators.trend.alma import alma
from indicators.volatility.atr import atr


class StrongTrendV147(TimeSeriesStrategy):
    """
    策略简介：High-Low Spread 高低价差估计 + ALMA 趋势方向确认的趋势策略。

    使用指标：
    - HL Spread(20): 高低价差估计，低值表示价差收敛（蓄力）
    - ALMA(9, 0.85, 6): Arnaud Legoux 移动均线，价格在上方确认多头
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - HL Spread < spread_threshold（价差收敛，准备突破）
    - 价格 > ALMA（多头趋势确认）

    出场条件：
    - ATR 追踪止损触发
    - 价格 < ALMA 时趋势反转信号退出

    优点：价差收敛+趋势方向组合，捕捉收敛后的突破
    缺点：价差收敛不一定向上突破，需要ALMA确认方向
    """
    name = "strong_trend_v147"
    warmup = 60
    freq = "daily"

    spread_period: int = 20
    spread_threshold: float = 0.5
    alma_period: int = 9
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._spread = None
        self._alma = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._spread = hl_spread(highs, lows, closes, self.spread_period)
        self._alma = alma(closes, self.alma_period, offset=0.85, sigma=6)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        sp = self._spread[i]
        alma_val = self._alma[i]
        atr_val = self._atr[i]
        if np.isnan(sp) or np.isnan(alma_val) or np.isnan(atr_val):
            return

        # Stop loss (FIRST)
        if side == 1:
            self.highest = max(self.highest, price)
            trailing = self.highest - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Entry
        if side == 0 and sp < self.spread_threshold and price > alma_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and price < alma_val:
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
