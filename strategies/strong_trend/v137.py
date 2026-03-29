"""Strong Trend v137 — OI Velocity spike + McGinley Dynamic trend."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.oi_velocity import oi_velocity
from indicators.trend.mcginley import mcginley_dynamic
from indicators.volatility.atr import atr


class StrongTrendV137(TimeSeriesStrategy):
    """
    策略简介：OI Velocity 持仓变速突增 + McGinley Dynamic 自适应均线趋势策略。

    使用指标：
    - OI Velocity(5): 持仓量变化速度，正值突增表示资金快速流入
    - McGinley Dynamic(14): 自适应均线，自动调节速度跟踪趋势
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - OI Velocity > vel_threshold（持仓快速增加）
    - 价格 > McGinley Dynamic（多头趋势确认）

    出场条件：
    - ATR 追踪止损触发
    - 价格 < McGinley Dynamic 时趋势反转信号退出

    优点：OI 变速捕捉资金快速入场时刻，McGinley 自适应减少假信号
    缺点：OI Velocity 周期短(5)，噪音较大
    """
    name = "strong_trend_v137"
    warmup = 60
    freq = "daily"

    oi_vel_period: int = 5
    mcg_period: int = 14
    vel_threshold: float = 1.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._oi_vel = None
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
        oi = context.get_full_oi_array()

        self._oi_vel = oi_velocity(oi, self.oi_vel_period)
        self._mcg = mcginley_dynamic(closes, self.mcg_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        vel = self._oi_vel[i]
        mcg_val = self._mcg[i]
        atr_val = self._atr[i]
        if np.isnan(vel) or np.isnan(mcg_val) or np.isnan(atr_val):
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
        if side == 0 and vel > self.vel_threshold and price > mcg_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and price < mcg_val:
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
