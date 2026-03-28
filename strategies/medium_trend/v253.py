import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.ml.gradient_trend import gradient_signal
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV253(TimeSeriesStrategy):
    """
    策略简介：梯度趋势信号检测，通过滚动梯度计算捕捉趋势加速阶段。
    使用指标：Gradient Signal(20,5) + ATR
    进场条件：梯度信号持续为正且加速
    出场条件：ATR追踪止损 / 梯度信号转负
    优点：对趋势加速敏感，信号平滑
    缺点：横盘期信号模糊
    """
    name = "mt_v253"
    warmup = 80
    freq = "daily"

    grad_period: int = 20
    grad_smooth: int = 5
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._grad = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        self._grad = gradient_signal(closes, period=self.grad_period, smoothing=self.grad_smooth)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        grad_val = self._grad[i]
        if np.isnan(atr_val) or np.isnan(grad_val) or atr_val <= 0:
            return
        if i < 1:
            return
        grad_prev = self._grad[i - 1]
        if np.isnan(grad_prev):
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and grad_val > 0 and grad_val > grad_prev:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and grad_val < 0:
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
