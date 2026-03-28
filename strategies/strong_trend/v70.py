"""Strong Trend v70 — Normalized Volume + Keltner Channel."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.normalized_volume import normalized_volume
from indicators.trend.keltner import keltner
from indicators.volatility.atr import atr


class StrongTrendV70(TimeSeriesStrategy):
    """
    策略简介：Normalized Volume 标准化成交量 + Keltner Channel 通道突破策略。

    使用指标：
    - Normalized Volume(20): 成交量/均量比，>1.5 为放量
    - Keltner Channel(20, 10, 1.5): 上中下轨，价格突破上轨确认动量
    - ATR(14): 追踪止损

    进场条件（做多）：
    - Normalized Volume > nv_threshold（显著放量）
    - 价格突破 Keltner 上轨（通道突破）

    出场条件：
    - ATR 追踪止损
    - 价格跌回 Keltner 中轨下方（动量消退）

    优点：Normalized Volume 跨品种可比，Keltner 比 Bollinger 更稳定
    缺点：通道突破在假突破时可能被反复止损
    """
    name = "strong_trend_v70"
    warmup = 60
    freq = "daily"

    nv_period: int = 20
    nv_threshold: float = 1.5
    kc_ema: int = 20
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._nv = None
        self._kc_upper = None
        self._kc_mid = None
        self._kc_lower = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._nv = normalized_volume(volumes, period=self.nv_period)
        upper, mid, lower = keltner(highs, lows, closes,
                                     ema=self.kc_ema, atr=10, mult=1.5)
        self._kc_upper = upper
        self._kc_mid = mid
        self._kc_lower = lower
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        nv_val = self._nv[i]
        kc_upper = self._kc_upper[i]
        kc_mid = self._kc_mid[i]
        atr_val = self._atr[i]
        if np.isnan(nv_val) or np.isnan(kc_upper) or np.isnan(atr_val):
            return

        # Stop loss
        if side == 1:
            self.highest = max(self.highest, price)
            trailing = self.highest - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Entry: high volume + Keltner breakout
        if side == 0 and nv_val > self.nv_threshold and price > kc_upper:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: price falls back below mid band
        elif side == 1 and price < kc_mid:
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
        self.highest = 0.0
        self.stop_price = 0.0
