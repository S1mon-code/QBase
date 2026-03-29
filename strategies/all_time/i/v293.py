import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.ml.gradient_trend import gradient_signal
from indicators.volume.volume_weighted_rsi import volume_rsi


class AllTimeIV293(TimeSeriesStrategy):
    """
    策略简介：Gradient Signal + Volume RSI的30min自适应策略。
    使用指标：Gradient Signal(20,5)趋势梯度，Volume RSI(14)量能方向。
    进场条件：梯度>0且VRSI>60做多；梯度<0且VRSI<40做空。
    出场条件：ATR追踪止损 + 梯度反向退出。
    优点：梯度信号平滑度好，VRSI提供量能确认。
    缺点：梯度在震荡市可能频繁翻转。
    """
    name = "i_alltime_v293"
    warmup = 60
    freq = "30min"

    vrsi_long: float = 60.0
    vrsi_short: float = 40.0
    atr_stop_mult: float = 2.5

    def __init__(self):
        super().__init__()
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest = 0.0
        self.lowest = 999999.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()


        self._atr = atr(highs, lows, closes, period=14)
        self._grad = gradient_signal(closes, period=20, smoothing=5)
        self._vrsi = volume_rsi(closes, volumes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._grad[i]) or np.isnan(self._vrsi[i]) or np.isnan(atr_val):
            return

        if side == 1:
            self.highest = max(self.highest, price)
            if price <= self.highest - self.atr_stop_mult * atr_val:
                context.close_long()
                self._reset()
                return

        elif side == -1:
            self.lowest = min(self.lowest, price)
            if price >= self.lowest + self.atr_stop_mult * atr_val:
                context.close_short()
                self._reset()
                return

        if side == 0 and self._grad[i] > 0 and self._vrsi[i] > self.vrsi_long:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._grad[i] < 0 and self._vrsi[i] < self.vrsi_short:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._grad[i] < 0:
            context.close_long()
            self._reset()

        elif side == -1 and self._grad[i] > 0:
            context.close_short()
            self._reset()

    def _calc_lots(self, context, price, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_stop_mult * atr_val * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = price * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        max_lots = int(context.equity * 0.30 / margin)
        return max(1, min(risk_lots, max_lots))

    def _reset(self):
        self.entry_price = 0.0
        self.highest = 0.0
        self.lowest = 999999.0
