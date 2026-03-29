import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.microstructure.volume_imbalance import volume_imbalance
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV292(TimeSeriesStrategy):
    """
    策略简介：成交量不平衡+价格动量策略，买方主导时做多。
    使用指标：Volume Imbalance(20) + ATR
    进场条件：Volume Imbalance持续为正且价格上升
    出场条件：ATR追踪止损 / Volume Imbalance转负
    优点：直接度量买卖力量对比
    缺点：30min级别噪声较大
    """
    name = "mt_v292"
    warmup = 300
    freq = "30min"

    vi_period: int = 20
    slope_lookback: int = 5
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._vi = None
        self._atr = None
        self._closes = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        self._vi = volume_imbalance(closes, volumes, period=self.vi_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._closes = closes

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        vi_val = self._vi[i]
        if np.isnan(atr_val) or np.isnan(vi_val) or atr_val <= 0:
            return
        if i < self.slope_lookback:
            return
        price_prev = self._closes[i - self.slope_lookback]
        if np.isnan(price_prev):
            return
        price_rising = price > price_prev

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and vi_val > 0.3 and price_rising:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and vi_val < -0.2:
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
