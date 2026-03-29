"""Strong Trend v146 — Delivery Pressure + Supertrend direction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.delivery_pressure import delivery_pressure
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr


class StrongTrendV146(TimeSeriesStrategy):
    """
    策略简介：Delivery Pressure 交割压力 + Supertrend 趋势方向确认的趋势策略。

    使用指标：
    - Delivery Pressure(20): 交割压力指标，低值表示交割压力小，看涨
    - Supertrend(10, 3.0): 趋势方向，dir==1 为多头
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - Delivery Pressure < dp_threshold（交割压力低）
    - Supertrend 方向为多头（dir == 1）

    出场条件：
    - ATR 追踪止损触发
    - Supertrend 翻空时信号退出

    优点：交割压力反映实物市场供需，配合趋势方向信号可靠
    缺点：交割压力指标需要OI和日期数据，数据质量要求高
    """
    name = "strong_trend_v146"
    warmup = 60
    freq = "daily"

    dp_period: int = 20
    dp_threshold: float = 0.5
    st_period: int = 10
    st_mult: float = 3.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._dp = None
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
        oi = context.get_full_oi_array()
        datetimes = context.get_full_datetime_array()

        self._dp = delivery_pressure(oi, volumes, datetimes, self.dp_period)
        _, self._st_dir = supertrend(highs, lows, closes, self.st_period, self.st_mult)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        dp_val = self._dp[i]
        st_dir = self._st_dir[i]
        atr_val = self._atr[i]
        if np.isnan(dp_val) or np.isnan(st_dir) or np.isnan(atr_val):
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
        if side == 0 and dp_val < self.dp_threshold and st_dir == 1:
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
