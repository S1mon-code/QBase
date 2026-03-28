import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.microstructure.volume_concentration import volume_concentration
from indicators.trend.donchian import donchian
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV299(TimeSeriesStrategy):
    """
    策略简介：成交量集中度+Donchian突破，高集中度=机构参与。
    使用指标：Volume Concentration(20,0.2) + Donchian(20) + ATR
    进场条件：成交量集中度高且价格突破Donchian上轨
    出场条件：ATR追踪止损 / 价格跌破Donchian中轨
    优点：集中成交量=大资金参与突破
    缺点：集中度高也可能是恐慌抛售
    """
    name = "mt_v299"
    warmup = 300
    freq = "30min"

    vc_period: int = 20
    dc_period: int = 20
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._vc = None
        self._dc_upper = None
        self._dc_mid = None
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
        self._vc = volume_concentration(volumes, period=self.vc_period, top_pct=0.2)
        self._dc_upper, _, self._dc_mid = donchian(highs, lows, period=self.dc_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        vc_val = self._vc[i]
        dc_u = self._dc_upper[i]
        dc_m = self._dc_mid[i]
        if np.isnan(atr_val) or np.isnan(vc_val) or np.isnan(dc_u) or np.isnan(dc_m) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and vc_val > 0.5 and price >= dc_u:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and price < dc_m:
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
