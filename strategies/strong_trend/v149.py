"""Strong Trend v149 — Trade Clustering + Fractal Levels breakout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.microstructure.trade_clustering import trade_clustering
from indicators.trend.fractal import fractal_levels
from indicators.volatility.atr import atr


class StrongTrendV149(TimeSeriesStrategy):
    """
    策略简介：Trade Clustering 交易聚集度 + Fractal Levels 分形突破的趋势策略。

    使用指标：
    - Trade Clustering(20): 交易聚集度，高值表示交易集中爆发
    - Fractal Levels: 分形高低点，价格突破分形高点确认突破
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - Trade Clustering > cluster_threshold（交易聚集爆发）
    - 价格 > 最近分形高点（突破分形阻力位）

    出场条件：
    - ATR 追踪止损触发
    - 价格 < 最近分形低点时信号退出

    优点：聚集度+分形突破组合捕捉关键位置的突破
    缺点：分形高低点更新滞后，可能在突破后才确认
    """
    name = "strong_trend_v149"
    warmup = 60
    freq = "daily"

    cluster_period: int = 20
    cluster_threshold: float = 1.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._cluster = None
        self._frac_high = None
        self._frac_low = None
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

        self._cluster = trade_clustering(volumes, self.cluster_period)
        self._frac_high, self._frac_low = fractal_levels(highs, lows)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        cl = self._cluster[i]
        fh = self._frac_high[i]
        fl = self._frac_low[i]
        atr_val = self._atr[i]
        if np.isnan(cl) or np.isnan(fh) or np.isnan(fl) or np.isnan(atr_val):
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
        if side == 0 and cl > self.cluster_threshold and price > fh:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and price < fl:
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
