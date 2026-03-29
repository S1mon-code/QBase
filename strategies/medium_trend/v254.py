import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.regime.trend_persistence import trend_persistence
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV254(TimeSeriesStrategy):
    """
    策略简介：趋势持续性检测策略，利用自相关分析判断趋势是否具有惯性。
    使用指标：Trend Persistence(max_lag=20, period=60) + ATR
    进场条件：趋势持续性高于阈值且价格创近期新高
    出场条件：ATR追踪止损 / 趋势持续性崩塌
    优点：直接度量趋势强度的统计显著性
    缺点：需要较长回溯期，信号不频繁
    """
    name = "mt_v254"
    warmup = 120
    freq = "daily"

    persist_period: int = 60
    persist_threshold: float = 0.6
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._persist = None
        self._atr = None
        self._highs = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        self._persist = trend_persistence(closes, max_lag=20, period=self.persist_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._highs = highs

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        persist_val = self._persist[i]
        if np.isnan(atr_val) or np.isnan(persist_val) or atr_val <= 0:
            return
        if i < 20:
            return
        recent_high = np.nanmax(self._highs[max(0, i - 20):i])
        at_high = price >= recent_high

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and persist_val > self.persist_threshold and at_high:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and persist_val < 0.3:
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
