import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.trend.vidya import vidya
from indicators.regime.vol_regime_markov import vol_regime_simple


class AllTimeIV216(TimeSeriesStrategy):
    """
    策略简介：VIDYA方向 + Vol Regime过滤的日线自适应趋势策略。
    使用指标：VIDYA(14)自适应均线方向，Vol Regime Markov过滤。
    进场条件：VIDYA上升且Vol Regime=低波动做多；VIDYA下降且低波动做空。
    出场条件：ATR追踪止损 + VIDYA反转退出。
    优点：VIDYA根据波动率自适应调整平滑度。
    缺点：高波动环境下信号较少。
    """
    name = "i_alltime_v216"
    warmup = 60
    freq = "daily"

    vol_regime_max: float = 1.5
    atr_stop_mult: float = 4.5

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
        self._vidya = vidya(closes, period=14)
        self._vol_regime = vol_regime_simple(closes, period=60)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._vidya[i]) or np.isnan(self._vol_regime[i]) or np.isnan(atr_val):
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

        if side == 0 and self._vidya[i] > self._vidya[i-1] and self._vol_regime[i] < self.vol_regime_max:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._vidya[i] < self._vidya[i-1] and self._vol_regime[i] < self.vol_regime_max:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._vidya[i] < self._vidya[i-1]:
            context.close_long()
            self._reset()

        elif side == -1 and self._vidya[i] > self._vidya[i-1]:
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
