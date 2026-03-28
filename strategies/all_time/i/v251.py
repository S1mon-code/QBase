import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.volatility.bollinger import bollinger_bands
from indicators.volume.volume_oscillator import volume_oscillator


class AllTimeIV251(TimeSeriesStrategy):
    """
    策略简介：BB收缩后突破 + Volume Oscillator确认的1h策略。
    使用指标：BB(20,2.0)宽度收缩后方向突破，VO(5,20)确认放量。
    进场条件：BB宽度<阈值后价格突破上轨且VO>0做多；突破下轨且VO>0做空。
    出场条件：ATR追踪止损 + 价格回到BB中轨。
    优点：收缩后突破成功率较高。
    缺点：需要等待收缩条件满足，信号较少。
    """
    name = "i_alltime_v251"
    warmup = 60
    freq = "1h"

    width_threshold: float = 0.03
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
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(closes, period=20, std=2.0)
        self._bb_width = (self._bb_upper - self._bb_lower) / np.where(closes > 0, closes, 1.0)
        self._vo = volume_oscillator(volumes, fast=5, slow=20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._bb_upper[i]) or np.isnan(self._bb_lower[i]) or np.isnan(self._vo[i]) or np.isnan(atr_val):
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

        if side == 0 and self._bb_width[i-1] < self.width_threshold and price > self._bb_upper[i] and self._vo[i] > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._bb_width[i-1] < self.width_threshold and price < self._bb_lower[i] and self._vo[i] > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and price < self._bb_mid[i]:
            context.close_long()
            self._reset()

        elif side == -1 and price > self._bb_mid[i]:
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
