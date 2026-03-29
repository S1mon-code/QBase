import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.ml.bayesian_trend import bayesian_online_trend
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV255(TimeSeriesStrategy):
    """
    策略简介：贝叶斯在线变点检测趋势策略，捕捉结构性趋势启动。
    使用指标：Bayesian Online Trend(hazard=0.01) + ATR
    进场条件：贝叶斯趋势信号从低转高，概率超过阈值
    出场条件：ATR追踪止损 / 趋势信号衰减
    优点：自适应检测趋势起点，无固定窗口
    缺点：计算复杂度较高，hazard率需调优
    """
    name = "mt_v255"
    warmup = 120
    freq = "daily"

    hazard_rate: float = 0.01
    entry_threshold: float = 0.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._bayes = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        self._bayes = bayesian_online_trend(closes, hazard_rate=self.hazard_rate)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        bayes_val = self._bayes[i]
        if np.isnan(atr_val) or np.isnan(bayes_val) or atr_val <= 0:
            return
        if i < 1:
            return
        bayes_prev = self._bayes[i - 1]
        if np.isnan(bayes_prev):
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and bayes_prev < self.entry_threshold and bayes_val >= self.entry_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and bayes_val < 0.2:
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
