"""Strong Trend v65 — Buying/Selling Pressure + EMA Ribbon."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.buying_selling_pressure import buying_selling_pressure
from indicators.trend.ema_ribbon import ema_ribbon_signal
from indicators.volatility.atr import atr


class StrongTrendV65(TimeSeriesStrategy):
    """
    策略简介：Buying/Selling Pressure 买卖压力 + EMA Ribbon 趋势带策略。

    使用指标：
    - Buying/Selling Pressure(14): 买卖双方压力差，>0 买方占优
    - EMA Ribbon Signal([8,13,21,34,55]): 多条EMA排列信号，>0 多头排列
    - ATR(14): 追踪止损

    进场条件（做多）：
    - Buying Pressure > Selling Pressure（买方力量占优）
    - EMA Ribbon Signal > ribbon_threshold（多头排列程度强）

    出场条件：
    - ATR 追踪止损
    - EMA Ribbon Signal < 0（多头排列瓦解）

    优点：Pressure 直接量化买卖力量对比，EMA Ribbon 多维度确认趋势
    缺点：EMA Ribbon 在趋势末期仍可能显示多头排列
    """
    name = "strong_trend_v65"
    warmup = 60
    freq = "daily"

    pressure_period: int = 14
    ribbon_threshold: float = 0.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._pressure = None
        self._ribbon = None
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

        self._pressure = buying_selling_pressure(highs, lows, closes, volumes,
                                                  period=self.pressure_period)
        self._ribbon = ema_ribbon_signal(closes, periods=[8, 13, 21, 34, 55])
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        press = self._pressure[i]
        ribbon = self._ribbon[i]
        atr_val = self._atr[i]
        if np.isnan(press) or np.isnan(ribbon) or np.isnan(atr_val):
            return

        # Stop loss
        if side == 1:
            self.highest = max(self.highest, price)
            trailing = self.highest - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Entry: buying pressure dominant + strong ribbon alignment
        if side == 0 and press > 0 and ribbon > self.ribbon_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: ribbon collapsed
        elif side == 1 and ribbon < 0:
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
