"""Strong Trend v144 — Warehouse Proxy (inventory proxy) + ROC momentum."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.warehouse_proxy import inventory_proxy
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr


class StrongTrendV144(TimeSeriesStrategy):
    """
    策略简介：Warehouse Proxy 库存代理指标（低库存看涨）+ ROC 动量确认的趋势策略。

    使用指标：
    - Inventory Proxy(40): 库存代理，低值/下降表示库存紧张，有利于价格上涨
    - ROC(12): 动量变化率，正值确认上涨趋势
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - Inventory Proxy 下降（当前 < 前一日，库存减少）
    - ROC > roc_threshold（正向动量确认）

    出场条件：
    - ATR 追踪止损触发
    - ROC < 0 时动量反转信号退出

    优点：库存下降+价格上涨的逻辑关系清晰（供给驱动）
    缺点：库存代理基于OI间接推算，精度有限
    """
    name = "strong_trend_v144"
    warmup = 60
    freq = "daily"

    inv_period: int = 40
    roc_period: int = 12
    roc_threshold: float = 3.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._inv = None
        self._roc = None
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
        oi = context.get_full_oi_array()

        self._inv = inventory_proxy(closes, oi, volumes, self.inv_period)
        self._roc = rate_of_change(closes, self.roc_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if i < 1:
            return

        inv_val = self._inv[i]
        inv_prev = self._inv[i - 1]
        roc_val = self._roc[i]
        atr_val = self._atr[i]
        if np.isnan(inv_val) or np.isnan(inv_prev) or np.isnan(roc_val) or np.isnan(atr_val):
            return

        inv_declining = inv_val < inv_prev

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
        if side == 0 and inv_declining and roc_val > self.roc_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and roc_val < 0:
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
