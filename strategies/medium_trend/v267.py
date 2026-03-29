import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.trend.hma import hma
from indicators.volatility.bollinger import bollinger_bands
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV267(TimeSeriesStrategy):
    """
    策略简介：HMA趋势方向+Bollinger突破组合。
    使用指标：HMA(20) + Bollinger Bands(20,2.0) + ATR
    进场条件：HMA上升且价格突破Bollinger上轨
    出场条件：ATR追踪止损 / 价格跌回Bollinger中轨下方
    优点：HMA低延迟+BB突破动能确认
    缺点：BB假突破在横盘期频繁
    """
    name = "mt_v267"
    warmup = 200
    freq = "4h"

    hma_period: int = 20
    bb_period: int = 20
    atr_trail_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._hma = None
        self._bb_upper = None
        self._bb_mid = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        self._hma = hma(closes, period=self.hma_period)
        self._bb_upper, self._bb_mid, _ = bollinger_bands(closes, period=self.bb_period, std=2.0)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        hma_val = self._hma[i]
        bb_u = self._bb_upper[i]
        bb_m = self._bb_mid[i]
        if np.isnan(atr_val) or np.isnan(hma_val) or np.isnan(bb_u) or np.isnan(bb_m) or atr_val <= 0:
            return
        if i < 1:
            return
        hma_prev = self._hma[i - 1]
        if np.isnan(hma_prev):
            return
        hma_rising = hma_val > hma_prev

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and hma_rising and price > bb_u:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and price < bb_m:
            context.close_long()
            self._reset()

    def _calc_lots(self, context, price, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_trail_mult * atr_val * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = price * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        return max(1, min(risk_lots, int(context.equity * 0.30 / margin)))

    def _reset(self):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
