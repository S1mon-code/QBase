import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.volume.oi_divergence import oi_divergence
from indicators.volume.volume_spike import volume_spike


class AllTimeIV261(TimeSeriesStrategy):
    """
    策略简介：OI Divergence + Volume Spike的1h量仓驱动策略。
    使用指标：OI Divergence(20)价格-持仓背离，Volume Spike(20,2.0)放量。
    进场条件：OI背离>0且放量做多；OI背离<0且放量做空。
    出场条件：ATR追踪止损 + 背离信号消失。
    优点：OI背离是期货市场特有的高质量信号。
    缺点：OI数据质量影响信号可靠性。
    """
    name = "i_alltime_v261"
    warmup = 60
    freq = "1h"

    div_threshold: float = 0.5
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
        self._oi_div = oi_divergence(closes, oi_arr, period=20)
        self._vol_spike = volume_spike(volumes, 20, 2.0)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._oi_div[i]) or np.isnan(self._vol_spike[i]) or np.isnan(atr_val):
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

        if side == 0 and self._oi_div[i] > self.div_threshold and self._vol_spike[i] > 0.5:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._oi_div[i] < -self.div_threshold and self._vol_spike[i] > 0.5:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._oi_div[i] < 0:
            context.close_long()
            self._reset()

        elif side == -1 and self._oi_div[i] > 0:
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
