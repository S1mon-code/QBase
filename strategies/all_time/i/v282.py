import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.regime.regime_score import composite_regime
from indicators.momentum.macd import macd


class AllTimeIV282(TimeSeriesStrategy):
    """
    策略简介：Composite Regime投票 + MACD方向的30min自适应策略。
    使用指标：Composite Regime Score(20)行情状态，MACD(12,26,9)方向。
    进场条件：Regime>0.3且MACD柱>0做多；Regime>0.3且MACD柱<0做空。
    出场条件：ATR追踪止损 + MACD柱反向退出。
    优点：Regime综合评估避免错误模式。
    缺点：Regime计算复杂度高。
    """
    name = "i_alltime_v282"
    warmup = 60
    freq = "30min"

    regime_threshold: float = 0.3
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
        self._regime = composite_regime(closes, highs, lows, period=20)
        _, _, self._macd_hist = macd(closes, fast=12, slow=26, signal=9)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        atr_val = self._atr[i]
        if np.isnan(self._regime[i]) or np.isnan(self._macd_hist[i]) or np.isnan(atr_val):
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

        if side == 0 and self._regime[i] > self.regime_threshold and self._macd_hist[i] > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price

        elif side == 0 and self._regime[i] > self.regime_threshold and self._macd_hist[i] < 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.sell(lot_size)
                self.entry_price = price
                self.lowest = price

        elif side == 1 and self._macd_hist[i] < 0:
            context.close_long()
            self._reset()

        elif side == -1 and self._macd_hist[i] > 0:
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
