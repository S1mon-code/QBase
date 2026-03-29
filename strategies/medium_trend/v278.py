import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.structure.position_crowding import position_crowding
from indicators.trend.adx import adx
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV278(TimeSeriesStrategy):
    """
    策略简介：持仓拥挤度+ADX趋势强度，低拥挤+强趋势=安全做多。
    使用指标：Position Crowding(60) + ADX(14) + ATR
    进场条件：拥挤度低（<0.5）且ADX>25
    出场条件：ATR追踪止损 / 拥挤度过高或ADX衰减
    优点：避免拥挤交易的反转风险
    缺点：拥挤度指标滞后
    """
    name = "mt_v278"
    warmup = 300
    freq = "1h"

    crowd_period: int = 60
    adx_threshold: int = 25
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._crowd = None
        self._adx = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()
        self._crowd = position_crowding(closes, oi, volumes, period=self.crowd_period)
        self._adx = adx(highs, lows, closes, period=14)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        crowd_val = self._crowd[i]
        adx_val = self._adx[i]
        if np.isnan(atr_val) or np.isnan(crowd_val) or np.isnan(adx_val) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and crowd_val < 0.5 and adx_val > self.adx_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and (crowd_val > 0.8 or adx_val < 18):
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
