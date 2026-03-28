import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.momentum.ppo import ppo
from indicators.volume.twiggs import twiggs_money_flow
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV268(TimeSeriesStrategy):
    """
    策略简介：PPO动量+Twiggs Money Flow资金流组合。
    使用指标：PPO(12,26,9) + Twiggs Money Flow(21) + ATR
    进场条件：PPO柱状图从负转正且TMF>0
    出场条件：ATR追踪止损 / PPO柱状图持续为负3根K线
    优点：规范化动量+机构级资金流
    缺点：TMF滞后于价格
    """
    name = "mt_v268"
    warmup = 200
    freq = "4h"

    ppo_fast: int = 12
    ppo_slow: int = 26
    tmf_period: int = 21
    atr_trail_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._ppo_line = None
        self._ppo_signal = None
        self._ppo_hist = None
        self._tmf = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
        self.neg_hist_count = 0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        self._ppo_line, self._ppo_signal, self._ppo_hist = ppo(closes, fast=self.ppo_fast, slow=self.ppo_slow, signal=9)
        self._tmf = twiggs_money_flow(highs, lows, closes, volumes, period=self.tmf_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        hist_val = self._ppo_hist[i]
        tmf_val = self._tmf[i]
        if np.isnan(atr_val) or np.isnan(hist_val) or np.isnan(tmf_val) or atr_val <= 0:
            return
        if i < 1:
            return
        hist_prev = self._ppo_hist[i - 1]
        if np.isnan(hist_prev):
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return
            if hist_val < 0:
                self.neg_hist_count += 1
            else:
                self.neg_hist_count = 0

        if side == 0 and hist_prev < 0 and hist_val >= 0 and tmf_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
                self.neg_hist_count = 0
        elif side == 1 and self.neg_hist_count >= 3:
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
        self.neg_hist_count = 0
