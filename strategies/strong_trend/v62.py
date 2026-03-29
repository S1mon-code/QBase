"""Strong Trend v62 — Demand Index + McGinley Dynamic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.demand_index import demand_index
from indicators.trend.mcginley import mcginley_dynamic
from indicators.volatility.atr import atr


class StrongTrendV62(TimeSeriesStrategy):
    """
    策略简介：Demand Index 需求指标 + McGinley Dynamic 自适应均线策略。

    使用指标：
    - Demand Index(14): 衡量买卖压力，>0 需求大于供给
    - McGinley Dynamic(14): 自适应移动平均，自动调节速度
    - ATR(14): 追踪止损

    进场条件（做多）：
    - Demand Index > di_threshold（买方需求显著）
    - 价格 > McGinley Dynamic（价格在自适应均线上方）

    出场条件：
    - ATR 追踪止损
    - 价格跌破 McGinley Dynamic 且 Demand Index < 0

    优点：McGinley 自适应速度减少假穿越，Demand Index 直接量化供需
    缺点：Demand Index 在低成交量品种可能不稳定
    """
    name = "strong_trend_v62"
    warmup = 60
    freq = "daily"

    di_period: int = 14
    mcg_period: int = 14
    di_threshold: float = 1.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._di = None
        self._mcg = None
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

        self._di = demand_index(highs, lows, closes, volumes, period=self.di_period)
        self._mcg = mcginley_dynamic(closes, period=self.mcg_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        di_val = self._di[i]
        mcg_val = self._mcg[i]
        atr_val = self._atr[i]
        if np.isnan(di_val) or np.isnan(mcg_val) or np.isnan(atr_val):
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

        # Entry: strong demand + price above adaptive MA
        if side == 0 and di_val > self.di_threshold and price > mcg_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: price below MA + demand gone
        elif side == 1 and price < mcg_val and di_val < 0:
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
