"""Strong Trend v131 — OI-Price Regime + Force Index confirmation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.oi_price_regime import oi_price_regime
from indicators.volume.force_index import force_index
from indicators.volatility.atr import atr


class StrongTrendV131(TimeSeriesStrategy):
    """
    策略简介：OI-Price Regime 持仓价格状态 + Force Index 量价力度确认的趋势策略。

    使用指标：
    - OI-Price Regime(20): 持仓与价格关系状态，正值=价涨仓增（多头趋势）
    - Force Index(13): 量价力度指标，正值表示多头力量
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - OI-Price Regime > 0（价涨仓增，多头趋势格局）
    - Force Index > 0（多头力量确认）

    出场条件：
    - ATR 追踪止损触发
    - Force Index < 0 时力度反转信号退出

    优点：OI-Price 状态直接反映资金与价格的共振
    缺点：OI数据滞后一天，可能错过日内快速变化
    """
    name = "strong_trend_v131"
    warmup = 60
    freq = "daily"

    regime_period: int = 20
    force_period: int = 13
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._regime = None
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
        oi = context.get_full_oi_array()

        self._regime = oi_price_regime(closes, oi, self.regime_period)
        self._force = force_index(closes, volumes, self.force_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        reg = self._regime[i]
        fi = self._force[i]
        atr_val = self._atr[i]
        if np.isnan(reg) or np.isnan(fi) or np.isnan(atr_val):
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
        if side == 0 and reg > 0 and fi > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and fi < 0:
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
