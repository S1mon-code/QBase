"""Strong Trend v145 — PVT Strength + Chandelier Exit trailing stop."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.price_volume_trend_strength import pvt_strength
from indicators.volatility.chandelier_exit import chandelier_exit
from indicators.volatility.atr import atr


class StrongTrendV145(TimeSeriesStrategy):
    """
    策略简介：PVT Strength 量价趋势强度 + Chandelier Exit 吊灯止损的趋势策略。

    使用指标：
    - PVT Strength(20): 量价趋势强度，正值且上升表示量价共振上涨
    - Chandelier Exit(22, 3.0): 吊灯止损线，提供趋势跟踪止损
    - ATR(14): 备用追踪止损距离

    进场条件（做多）：
    - PVT Strength > pvt_threshold（量价趋势强度高）
    - 价格 > Chandelier Long Exit（在吊灯止损上方）

    出场条件：
    - ATR 追踪止损触发
    - 价格 < Chandelier Long Exit 时趋势结束信号退出

    优点：PVT 综合量价信息，Chandelier 提供动态止损
    缺点：两个指标都偏滞后，趋势反转时可能回吐较多利润
    """
    name = "strong_trend_v145"
    warmup = 60
    freq = "daily"

    pvt_period: int = 20
    pvt_threshold: float = 0.5
    chand_period: int = 22
    chand_mult: float = 3.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._pvt = None
        self._chand_long = None
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

        self._pvt = pvt_strength(closes, volumes, self.pvt_period)
        self._chand_long, _ = chandelier_exit(highs, lows, closes, self.chand_period, self.chand_mult)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        pvt_val = self._pvt[i]
        chand_val = self._chand_long[i]
        atr_val = self._atr[i]
        if np.isnan(pvt_val) or np.isnan(chand_val) or np.isnan(atr_val):
            return

        # Stop loss (FIRST)
        if side == 1:
            self.highest = max(self.highest, price)
            trailing = self.highest - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Entry
        if side == 0 and pvt_val > self.pvt_threshold and price > chand_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and price < chand_val:
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
