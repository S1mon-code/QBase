import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.regime.variance_ratio import variance_ratio_test
from indicators.trend.adx import adx
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV258(TimeSeriesStrategy):
    """
    策略简介：方差比检验+ADX双重趋势确认，VR>1说明趋势性强，ADX确认方向。
    使用指标：Variance Ratio Test(60,5) + ADX(14) + ATR
    进场条件：VR > 1.1 且 ADX > 25
    出场条件：ATR追踪止损 / ADX跌破20
    优点：双重统计确认，假信号少
    缺点：进场偏晚，可能错过启动阶段
    """
    name = "mt_v258"
    warmup = 120
    freq = "daily"

    vr_threshold: float = 1.1
    adx_threshold: int = 25
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._vr = None
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
        self._vr = variance_ratio_test(closes, period=60, holding=5)
        self._adx = adx(highs, lows, closes, period=14)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        vr_val = self._vr[i]
        adx_val = self._adx[i]
        if np.isnan(atr_val) or np.isnan(vr_val) or np.isnan(adx_val) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and vr_val > self.vr_threshold and adx_val > self.adx_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and adx_val < 20:
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
