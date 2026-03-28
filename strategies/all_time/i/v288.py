import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.regime.fractal_dimension import fractal_dim
from indicators.momentum.fisher_transform import fisher_transform


class AllTimeIV288(TimeSeriesStrategy):
    """
    策略简介：Fractal Dimension自适应 + Fisher Transform的30min策略。
    使用指标：FD(60)区分趋势/随机，Fisher Transform(10)方向。
    进场条件：FD<1.4时Fisher>1做多（趋势）；FD>1.6时Fisher<-1做多（回归）。
    出场条件：ATR追踪止损 + Fisher回归0。
    优点：FD理论基础好，Fisher提供清晰信号。
    缺点：FD计算窗口大导致滞后。
    """
    name = "i_alltime_v288"
    warmup = 60
    freq = "30min"

    fd_trend: float = 1.4
    fd_range: float = 1.6
    fisher_threshold: float = 1.0
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
        self._fd = fractal_dim(closes, period=60)
        self._fisher, _ = fisher_transform(highs, lows, period=10)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._fd[i]) or np.isnan(self._fisher[i]) or np.isnan(atr_val):
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

        if side == 0 and (self._fd[i] < self.fd_trend and self._fisher[i] > self.fisher_threshold) or (self._fd[i] > self.fd_range and self._fisher[i] < -self.fisher_threshold):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and (self._fd[i] < self.fd_trend and self._fisher[i] < -self.fisher_threshold) or (self._fd[i] > self.fd_range and self._fisher[i] > self.fisher_threshold):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and abs(self._fisher[i]) < 0.5:
            context.close_long()
            self._reset()

        elif side == -1 and abs(self._fisher[i]) < 0.5:
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
