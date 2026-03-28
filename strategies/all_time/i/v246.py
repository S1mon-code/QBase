import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.volatility.vol_ratio import volatility_ratio
from indicators.volume.klinger import klinger


class AllTimeIV246(TimeSeriesStrategy):
    """
    策略简介：Volatility Ratio突破 + Klinger确认的1h策略。
    使用指标：Volatility Ratio(14)波动突破，Klinger(34,55,13)量能。
    进场条件：VR>1.5且KVO>Signal做多；VR>1.5且KVO<Signal做空。
    出场条件：ATR追踪止损 + KVO/Signal交叉反转。
    优点：VR检测波动率突破，Klinger确认方向。
    缺点：VR阈值需要根据品种调整。
    """
    name = "i_alltime_v246"
    warmup = 60
    freq = "1h"

    vr_threshold: float = 1.5
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
        self._vr = volatility_ratio(highs, lows, closes, period=14)
        self._kvo, self._kvo_signal = klinger(highs, lows, closes, volumes)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._vr[i]) or np.isnan(self._kvo[i]) or np.isnan(self._kvo_signal[i]) or np.isnan(atr_val):
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

        if side == 0 and self._vr[i] > self.vr_threshold and self._kvo[i] > self._kvo_signal[i]:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._vr[i] > self.vr_threshold and self._kvo[i] < self._kvo_signal[i]:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._kvo[i] < self._kvo_signal[i]:
            context.close_long()
            self._reset()

        elif side == -1 and self._kvo[i] > self._kvo_signal[i]:
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
