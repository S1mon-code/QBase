"""Strong Trend v143 — Net Positioning Proxy + Donchian Channel breakout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.net_positioning import net_positioning_proxy
from indicators.trend.donchian import donchian
from indicators.volatility.atr import atr


class StrongTrendV143(TimeSeriesStrategy):
    """
    策略简介：Net Positioning Proxy 净持仓代理 + Donchian 通道突破的趋势策略。

    使用指标：
    - Net Positioning Proxy(20): 净持仓方向代理，正值表示多头净持仓
    - Donchian Channel(20): 价格突破上轨确认趋势
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - Net Positioning > 0（多头净持仓）
    - 价格 > Donchian 上轨（价格突破确认）

    出场条件：
    - ATR 追踪止损触发
    - 价格 < Donchian 中线时信号退出

    优点：净持仓+通道突破经典组合，逻辑清晰
    缺点：Donchian 突破在震荡市会频繁触发假突破
    """
    name = "strong_trend_v143"
    warmup = 60
    freq = "daily"

    net_pos_period: int = 20
    donchian_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._net_pos = None
        self._dc_upper = None
        self._dc_mid = None
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

        self._net_pos = net_positioning_proxy(closes, oi, volumes, self.net_pos_period)
        self._dc_upper, _, self._dc_mid = donchian(highs, lows, self.donchian_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        np_val = self._net_pos[i]
        dc_u = self._dc_upper[i]
        dc_m = self._dc_mid[i]
        atr_val = self._atr[i]
        if np.isnan(np_val) or np.isnan(dc_u) or np.isnan(dc_m) or np.isnan(atr_val):
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
        if side == 0 and np_val > 0 and price > dc_u:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
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
        self.highest = 0.0
        self.stop_price = 0.0
