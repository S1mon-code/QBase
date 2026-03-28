import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.structure.oi_relative_strength import oi_relative_strength
from indicators.momentum.ppo import ppo


class AllTimeIV276(TimeSeriesStrategy):
    """
    策略简介：OI Relative Strength + PPO方向的1h策略。
    使用指标：OI Relative Strength(20)持仓相对强度，PPO(12,26,9)方向。
    进场条件：OI RS升高且PPO>Signal做多；OI RS升高且PPO<Signal做空。
    出场条件：ATR追踪止损 + PPO/Signal交叉反转。
    优点：OI相对强度过滤低质量信号。
    缺点：两个指标同时满足可能过严。
    """
    name = "i_alltime_v276"
    warmup = 60
    freq = "1h"

    atr_stop_mult: float = 3.0

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
        oi_arr = context.get_full_oi_array()
        self._oi_rs = oi_relative_strength(oi_arr, volumes, period=20)
        self._ppo, self._ppo_signal, _ = ppo(closes, fast=12, slow=26, signal=9)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._oi_rs[i]) or np.isnan(self._ppo[i]) or np.isnan(self._ppo_signal[i]) or np.isnan(atr_val):
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

        if side == 0 and self._oi_rs[i] > self._oi_rs[i-1] and self._ppo[i] > self._ppo_signal[i]:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._oi_rs[i] > self._oi_rs[i-1] and self._ppo[i] < self._ppo_signal[i]:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._ppo[i] < self._ppo_signal[i]:
            context.close_long()
            self._reset()

        elif side == -1 and self._ppo[i] > self._ppo_signal[i]:
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
