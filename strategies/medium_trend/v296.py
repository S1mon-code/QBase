import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.microstructure.range_to_volume import range_to_volume
from indicators.trend.ema import ema
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV296(TimeSeriesStrategy):
    """
    策略简介：Range-to-Volume效率+EMA趋势过滤。
    使用指标：Range to Volume(20) + EMA(30) + ATR
    进场条件：RTV下降（高效率趋势）且价格在EMA上方
    出场条件：ATR追踪止损 / 价格跌破EMA
    优点：度量每单位成交量的价格移动效率
    缺点：RTV在低量时不稳定
    """
    name = "mt_v296"
    warmup = 300
    freq = "30min"

    rtv_period: int = 20
    ema_period: int = 30
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._rtv = None
        self._ema = None
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
        self._rtv = range_to_volume(highs, lows, volumes, period=self.rtv_period)
        self._ema = ema(closes, period=self.ema_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        rtv_val = self._rtv[i]
        ema_val = self._ema[i]
        if np.isnan(atr_val) or np.isnan(rtv_val) or np.isnan(ema_val) or atr_val <= 0:
            return
        if i < 5:
            return
        rtv_prev = self._rtv[i - 5]
        if np.isnan(rtv_prev):
            return
        rtv_efficient = rtv_val > rtv_prev  # Higher range per volume = more efficient

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and rtv_efficient and price > ema_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and price < ema_val:
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
