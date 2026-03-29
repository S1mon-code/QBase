import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.structure.net_positioning import net_positioning_proxy
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV277(TimeSeriesStrategy):
    """
    策略简介：净持仓代理指标策略，捕捉多头持仓增加的趋势阶段。
    使用指标：Net Positioning Proxy(20) + ATR
    进场条件：净持仓代理值持续为正且增长
    出场条件：ATR追踪止损 / 净持仓转负
    优点：从量价关系推断持仓方向
    缺点：代理指标精度有限
    """
    name = "mt_v277"
    warmup = 200
    freq = "1h"

    np_period: int = 20
    slope_lookback: int = 5
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._np = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()
        self._np = net_positioning_proxy(closes, oi, volumes, period=self.np_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        np_val = self._np[i]
        if np.isnan(atr_val) or np.isnan(np_val) or atr_val <= 0:
            return
        if i < self.slope_lookback:
            return
        np_prev = self._np[i - self.slope_lookback]
        if np.isnan(np_prev):
            return
        np_growing = np_val > 0 and np_val > np_prev

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and np_growing:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and np_val < 0:
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
