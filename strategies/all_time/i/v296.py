import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.trend.supertrend import supertrend
from indicators.trend.psar import psar
from indicators.trend.aroon import aroon


class AllTimeIV296(TimeSeriesStrategy):
    """
    策略简介：多指标投票 (Supertrend+PSAR+Aroon) 的30min策略。
    使用指标：Supertrend(10,2.5)，PSAR方向，Aroon Oscillator(25)投票。
    进场条件：3票中>=2票看多做多；>=2票看空做空。
    出场条件：ATR追踪止损 + 投票翻转退出。
    优点：三个趋势指标投票，减少假信号。
    缺点：趋势指标同质化高。
    """
    name = "i_alltime_v296"
    warmup = 60
    freq = "30min"

    atr_stop_mult: float = 2.5

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
        _, self._st_dir = supertrend(highs, lows, closes, 10, 2.5)
        _, self._psar_dir = psar(highs, lows)
        _, _, self._aroon_osc = aroon(highs, lows, period=25)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._st_dir[i]) or np.isnan(self._psar_dir[i]) or np.isnan(self._aroon_osc[i]) or np.isnan(atr_val):
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

        if side == 0 and (int(self._st_dir[i] == 1) + int(self._psar_dir[i] == 1) + int(self._aroon_osc[i] > 0)) >= 2:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and (int(self._st_dir[i] == -1) + int(self._psar_dir[i] == -1) + int(self._aroon_osc[i] < 0)) >= 2:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and (int(self._st_dir[i] == -1) + int(self._psar_dir[i] == -1) + int(self._aroon_osc[i] < 0)) >= 2:
            context.close_long()
            self._reset()

        elif side == -1 and (int(self._st_dir[i] == 1) + int(self._psar_dir[i] == 1) + int(self._aroon_osc[i] > 0)) >= 2:
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
