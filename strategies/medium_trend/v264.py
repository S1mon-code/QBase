import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.trend.vortex import vortex
from indicators.momentum.rsi import rsi
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV264(TimeSeriesStrategy):
    """
    策略简介：Vortex趋势检测+RSI动量过滤，VI+>VI-确认上升趋势。
    使用指标：Vortex(14) + RSI(14) + ATR
    进场条件：VI+上穿VI-且RSI>50
    出场条件：ATR追踪止损 / VI+下穿VI-
    优点：Vortex对趋势启动敏感，RSI过滤超买
    缺点：窄幅震荡时Vortex频繁交叉
    """
    name = "mt_v264"
    warmup = 200
    freq = "4h"

    vortex_period: int = 14
    rsi_period: int = 14
    atr_trail_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._vi_plus = None
        self._vi_minus = None
        self._rsi = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        self._vi_plus, self._vi_minus = vortex(highs, lows, closes, period=self.vortex_period)
        self._rsi = rsi(closes, period=self.rsi_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        vip = self._vi_plus[i]
        vim = self._vi_minus[i]
        rsi_val = self._rsi[i]
        if np.isnan(atr_val) or np.isnan(vip) or np.isnan(vim) or np.isnan(rsi_val) or atr_val <= 0:
            return
        if i < 1:
            return
        vip_prev = self._vi_plus[i - 1]
        vim_prev = self._vi_minus[i - 1]
        if np.isnan(vip_prev) or np.isnan(vim_prev):
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and vip_prev < vim_prev and vip >= vim and rsi_val > 50:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and vip < vim:
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
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
