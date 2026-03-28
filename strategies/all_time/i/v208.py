import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.trend.aroon import aroon
from indicators.regime.variance_ratio import variance_ratio_test


class AllTimeIV208(TimeSeriesStrategy):
    """
    策略简介：Aroon方向 + Variance Ratio确认的日线趋势策略。
    使用指标：Aroon(25)振荡器方向，Variance Ratio测试随机性。
    进场条件：Aroon Osc>50且VR>1.1做多；Osc<-50且VR>1.1做空。
    出场条件：ATR追踪止损 + Aroon反转退出。
    优点：VR有效识别随机游走vs趋势。
    缺点：Aroon在窄幅震荡时可能频繁翻转。
    """
    name = "i_alltime_v208"
    warmup = 60
    freq = "daily"

    aroon_threshold: float = 50.0
    vr_threshold: float = 1.1
    atr_stop_mult: float = 4.0

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

        self._atr = atr(highs, lows, closes, period=14)
        _, _, self._aroon_osc = aroon(highs, lows, period=25)
        self._vr = variance_ratio_test(closes, period=60, holding=5)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._aroon_osc[i]) or np.isnan(self._vr[i]) or np.isnan(atr_val):
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

        if side == 0 and self._aroon_osc[i] > self.aroon_threshold and self._vr[i] > self.vr_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._aroon_osc[i] < -self.aroon_threshold and self._vr[i] > self.vr_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._aroon_osc[i] < 0:
            context.close_long()
            self._reset()

        elif side == -1 and self._aroon_osc[i] > 0:
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
