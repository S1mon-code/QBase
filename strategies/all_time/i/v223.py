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
from indicators.momentum.cci import cci


class AllTimeIV223(TimeSeriesStrategy):
    """
    策略简介：BB均值回归 + CCI确认的4h策略。
    使用指标：BB(20,2.0)上下轨突破回归，CCI(20)确认。
    进场条件：价格触BB下轨且CCI<-100做多；触上轨且CCI>100做空。
    出场条件：固定ATR止损 + 价格回归BB中轨。
    优点：价格+振荡器双重确认。
    缺点：布林带突破可能是趋势启动而非回归。
    """
    name = "i_alltime_v223"
    warmup = 60
    freq = "4h"

    cci_threshold: float = 100.0
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
        self._cci = cci(highs, lows, closes, period=20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._bb_upper[i]) or np.isnan(self._bb_lower[i]) or np.isnan(self._cci[i]) or np.isnan(atr_val):
            return

        if side == 1:
            if price <= self.entry_price - self.atr_stop_mult * atr_val:
                context.close_long()
                self._reset()
                return

        elif side == -1:
            if price >= self.entry_price + self.atr_stop_mult * atr_val:
                context.close_short()
                self._reset()
                return

        if side == 0 and price < self._bb_lower[i] and self._cci[i] < -self.cci_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price

        elif side == 0 and price > self._bb_upper[i] and self._cci[i] > self.cci_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price

        elif side == 1 and price > self._bb_mid[i]:
            context.close_long()
            self._reset()

        elif side == -1 and price < self._bb_mid[i]:
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
