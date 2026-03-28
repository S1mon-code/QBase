"""Strong Trend v52 — VROC + Force Index volume acceleration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.vroc import vroc
from indicators.volume.force_index import force_index
from indicators.volatility.atr import atr


class StrongTrendV52(TimeSeriesStrategy):
    """
    策略简介：VROC + Force Index 成交量加速度策略。

    使用指标：
    - VROC(14): 成交量变化率，>50% 说明放量加速
    - Force Index(13): Elder Force Index，>0 说明多头力量占优
    - ATR(14): 追踪止损

    进场条件（做多）：
    - VROC > vroc_threshold（成交量加速上升）
    - Force Index > 0（多头力量正向）

    出场条件：
    - ATR 追踪止损
    - Force Index 连续转负时信号退出

    优点：Force Index 结合价格和成交量，比单纯成交量更灵敏
    缺点：VROC 波动大，可能产生噪音信号
    """
    name = "strong_trend_v52"
    warmup = 60
    freq = "daily"

    vroc_period: int = 14
    force_period: int = 13
    vroc_threshold: float = 50.0
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._vroc = None
        self._force = None
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

        self._vroc = vroc(volumes, period=self.vroc_period)
        self._force = force_index(closes, volumes, period=self.force_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        vroc_val = self._vroc[i]
        force_val = self._force[i]
        atr_val = self._atr[i]
        if np.isnan(vroc_val) or np.isnan(force_val) or np.isnan(atr_val):
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

        # Entry: volume accelerating + bullish force
        if side == 0 and vroc_val > self.vroc_threshold and force_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: force turning bearish
        elif side == 1 and force_val < 0:
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
