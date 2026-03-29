import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.seasonality.seasonal_decompose import seasonal_strength
from indicators.momentum.elder_force import elder_force_index
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV288(TimeSeriesStrategy):
    """
    策略简介：季节性分解强度+Elder Force Index组合。
    使用指标：Seasonal Strength(252) + Elder Force Index(13) + ATR
    进场条件：季节性强度高（趋势非季节主导）且Force Index>0
    出场条件：ATR追踪止损 / Force Index持续为负
    优点：区分趋势vs季节成分+力度确认
    缺点：季节性分解需要长数据
    """
    name = "mt_v288"
    warmup = 400
    freq = "1h"

    seasonal_period: int = 252
    fi_period: int = 13
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._seas_str = None
        self._fi = None
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
        self._seas_str = seasonal_strength(closes, period=self.seasonal_period)
        self._fi = elder_force_index(closes, volumes, period=self.fi_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        ss_val = self._seas_str[i]
        fi_val = self._fi[i]
        if np.isnan(atr_val) or np.isnan(ss_val) or np.isnan(fi_val) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # High seasonal strength means trend component dominates
        if side == 0 and ss_val > 0.5 and fi_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and fi_val < 0:
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
