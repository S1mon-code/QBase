import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.microstructure.high_low_spread import hl_spread
from indicators.volume.buying_selling_pressure import buying_selling_pressure
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV298(TimeSeriesStrategy):
    """
    策略简介：High-Low Spread+买卖压力组合，窄价差+买方压力做多。
    使用指标：HL Spread(20) + Buying Selling Pressure(14) + ATR
    进场条件：HL Spread收窄（波动率压缩）且买方压力>卖方压力
    出场条件：ATR追踪止损 / 卖方压力超过买方
    优点：波动率压缩后突破+买方主导
    缺点：价差收窄不一定导致向上突破
    """
    name = "mt_v298"
    warmup = 300
    freq = "30min"

    spread_period: int = 20
    bp_period: int = 14
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._spread = None
        self._bp = None
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
        self._spread = hl_spread(highs, lows, closes, period=self.spread_period)
        self._bp = buying_selling_pressure(highs, lows, closes, volumes, period=self.bp_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        sp_val = self._spread[i]
        bp_val = self._bp[i]
        if np.isnan(atr_val) or np.isnan(sp_val) or np.isnan(bp_val) or atr_val <= 0:
            return
        if i < 5:
            return
        sp_prev = self._spread[i - 5]
        if np.isnan(sp_prev):
            return
        spread_narrowing = sp_val < sp_prev

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and spread_narrowing and bp_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and bp_val < -0.3:
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
