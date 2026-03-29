"""Strong Trend v61 — OI Flow + Vortex Indicator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.oi_flow import oi_flow
from indicators.trend.vortex import vortex
from indicators.volatility.atr import atr


class StrongTrendV61(TimeSeriesStrategy):
    """
    策略简介：OI Flow 资金流向 + Vortex 趋势方向策略。

    使用指标：
    - OI Flow(14): 综合OI+价格+量的资金流指标，>0 看多
    - Vortex(14): VI+ > VI- 确认上升趋势方向
    - ATR(14): 追踪止损

    进场条件（做多）：
    - OI Flow > flow_threshold（资金净流入显著）
    - VI+ > VI-（上升趋势方向确认）

    出场条件：
    - ATR 追踪止损
    - VI- > VI+ 且 OI Flow < 0（趋势反转 + 资金流出）

    优点：OI Flow 综合三维信息，Vortex 对趋势方向敏感
    缺点：Vortex 在趋势转换期可能频繁交叉
    """
    name = "strong_trend_v61"
    warmup = 60
    freq = "daily"

    oi_flow_period: int = 14
    vortex_period: int = 14
    flow_threshold: float = 0.5
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._oi_flow = None
        self._vi_plus = None
        self._vi_minus = None
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

        self._oi_flow = oi_flow(closes, oi, volumes, period=self.oi_flow_period)
        vi_plus, vi_minus = vortex(highs, lows, closes, period=self.vortex_period)
        self._vi_plus = vi_plus
        self._vi_minus = vi_minus
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        flow = self._oi_flow[i]
        vip = self._vi_plus[i]
        vim = self._vi_minus[i]
        atr_val = self._atr[i]
        if np.isnan(flow) or np.isnan(vip) or np.isnan(vim) or np.isnan(atr_val):
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

        # Entry: OI flow positive + Vortex bullish
        if side == 0 and flow > self.flow_threshold and vip > vim:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: trend reversal + flow negative
        elif side == 1 and vim > vip and flow < 0:
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
