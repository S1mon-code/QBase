"""Strong Trend v140 — OI Divergence Enhanced + Vortex trend."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.oi_divergence_enhanced import oi_divergence_enhanced
from indicators.trend.vortex import vortex
from indicators.volatility.atr import atr


class StrongTrendV140(TimeSeriesStrategy):
    """
    策略简介：OI Divergence Enhanced 增强版OI背离 + Vortex 趋势方向确认的策略。

    使用指标：
    - OI Divergence Enhanced(20): 增强版持仓量背离，正值表示OI支持上涨
    - Vortex(14): 正负涡流指标，VI+ > VI- 确认多头趋势
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - OI Divergence Enhanced > 0（OI确认上涨趋势）
    - Vortex VI+ > VI-（多头涡流占优）

    出场条件：
    - ATR 追踪止损触发
    - Vortex VI- > VI+ 时趋势反转信号退出

    优点：OI背离增强版结合量价，Vortex 提供清晰方向判断
    缺点：两个指标在震荡市可能频繁交叉，产生假信号
    """
    name = "strong_trend_v140"
    warmup = 60
    freq = "daily"

    oi_div_period: int = 20
    vortex_period: int = 14
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._oi_div = None
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

        self._oi_div = oi_divergence_enhanced(closes, oi, volumes, self.oi_div_period)
        self._vi_plus, self._vi_minus = vortex(highs, lows, closes, self.vortex_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        div_val = self._oi_div[i]
        vip = self._vi_plus[i]
        vim = self._vi_minus[i]
        atr_val = self._atr[i]
        if np.isnan(div_val) or np.isnan(vip) or np.isnan(vim) or np.isnan(atr_val):
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
        if side == 0 and div_val > 0 and vip > vim:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and vim > vip:
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
