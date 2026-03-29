"""Strong Trend v127 — Kyle's Lambda (price impact) + Supertrend."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.microstructure.price_impact import kyle_lambda
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr


class StrongTrendV127(TimeSeriesStrategy):
    """
    策略简介：Kyle's Lambda 低价格冲击 + Supertrend 方向确认的趋势跟踪策略。

    使用指标：
    - Kyle's Lambda(60): 价格冲击系数，低值表示市场深度好
    - Supertrend(10, 3.0): 趋势方向判断
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - Kyle's Lambda < lambda_threshold（低价格冲击，市场深度好）
    - Supertrend 方向为多头（dir == 1）

    出场条件：
    - ATR 追踪止损触发
    - Supertrend 翻空时信号退出

    优点：价格冲击低时入场，执行滑点可控
    缺点：Kyle's Lambda 计算周期长(60)，信号滞后
    """
    name = "strong_trend_v127"
    warmup = 60
    freq = "daily"

    lambda_period: int = 60
    st_period: int = 10
    st_mult: float = 3.0
    lambda_threshold: float = 0.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._lambda = None
        self._st_dir = None
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

        self._lambda = kyle_lambda(closes, volumes, self.lambda_period)
        _, self._st_dir = supertrend(highs, lows, closes, self.st_period, self.st_mult)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        lam = self._lambda[i]
        st_dir = self._st_dir[i]
        atr_val = self._atr[i]
        if np.isnan(lam) or np.isnan(st_dir) or np.isnan(atr_val):
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
        if side == 0 and lam < self.lambda_threshold and st_dir == 1:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and st_dir == -1:
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
