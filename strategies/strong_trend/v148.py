"""Strong Trend v148 — Range to Volume ratio + OBV confirmation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.microstructure.range_to_volume import range_to_volume
from indicators.volume.obv import obv
from indicators.volatility.atr import atr


class StrongTrendV148(TimeSeriesStrategy):
    """
    策略简介：Range to Volume 振幅量比 + OBV 累积量能方向确认的趋势策略。

    使用指标：
    - Range to Volume(20): 振幅/成交量比，低值表示量大但振幅小（蓄力）
    - OBV: On-Balance Volume，上升确认量能流入方向
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - Range to Volume < rv_threshold（量大振幅小，蓄力中）
    - OBV 上升（当前 > 前一日，量能流入）

    出场条件：
    - ATR 追踪止损触发
    - OBV 连续下降时量能流出信号退出

    优点：蓄力阶段入场+OBV确认方向，捕捉放量突破前信号
    缺点：Range to Volume 在低波动期常态化低值，可能误判
    """
    name = "strong_trend_v148"
    warmup = 60
    freq = "daily"

    rv_period: int = 20
    rv_threshold: float = 0.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._rv = None
        self._obv = None
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

        self._rv = range_to_volume(highs, lows, volumes, self.rv_period)
        self._obv = obv(closes, volumes)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if i < 1:
            return

        rv_val = self._rv[i]
        obv_val = self._obv[i]
        obv_prev = self._obv[i - 1]
        atr_val = self._atr[i]
        if np.isnan(rv_val) or np.isnan(obv_val) or np.isnan(obv_prev) or np.isnan(atr_val):
            return

        obv_rising = obv_val > obv_prev

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
        if side == 0 and rv_val < self.rv_threshold and obv_rising:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit — OBV declining for 2 consecutive bars
        elif side == 1 and i >= 2:
            obv_prev2 = self._obv[i - 2]
            if not np.isnan(obv_prev2) and obv_val < obv_prev < obv_prev2:
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
