import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.seasonality.vol_seasonality import vol_seasonality
from indicators.momentum.tsmom import tsmom
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV285(TimeSeriesStrategy):
    """
    策略简介：波动率季节性+TSMOM趋势动量组合。
    使用指标：Vol Seasonality(20) + TSMOM(252,60) + ATR
    进场条件：波动率季节性低（稳定期）且TSMOM>0
    出场条件：ATR追踪止损 / TSMOM转负
    优点：低波季节+趋势共振
    缺点：TSMOM长周期可能错过短期机会
    """
    name = "mt_v285"
    warmup = 400
    freq = "1h"

    tsmom_lookback: int = 252
    tsmom_vol: int = 60
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._vol_seas = None
        self._tsmom = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        datetimes = context.get_full_datetime_array()
        self._vol_seas = vol_seasonality(closes, datetimes, vol_period=20)
        self._tsmom = tsmom(closes, lookback=self.tsmom_lookback, vol_lookback=self.tsmom_vol)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        vs_val = self._vol_seas[i]
        ts_val = self._tsmom[i]
        if np.isnan(atr_val) or np.isnan(vs_val) or np.isnan(ts_val) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Low vol seasonality (below median) + positive TSMOM
        if side == 0 and vs_val < 0.5 and ts_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and ts_val < 0:
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
