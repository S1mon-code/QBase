"""Strong Trend v134 — Position Crowding (low = uncrowded entry) + PPO."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.position_crowding import position_crowding
from indicators.momentum.ppo import ppo
from indicators.volatility.atr import atr


class StrongTrendV134(TimeSeriesStrategy):
    """
    策略简介：Position Crowding 低拥挤度（未拥挤进场）+ PPO 动量确认的趋势策略。

    使用指标：
    - Position Crowding(60): 持仓拥挤度，低值表示市场未拥挤
    - PPO(12, 26, 9): 百分比价格振荡器，正值确认多头动量
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - Position Crowding < crowd_threshold（未拥挤，安全入场）
    - PPO > 0（正向动量确认）

    出场条件：
    - ATR 追踪止损触发
    - PPO < -1.0 时动量反转信号退出

    优点：低拥挤度入场避免踩踏风险，PPO 提供清晰动量信号
    缺点：拥挤度计算周期长，可能在趋势已走较远时才确认低拥挤
    """
    name = "strong_trend_v134"
    warmup = 60
    freq = "daily"

    crowd_period: int = 60
    crowd_threshold: float = 0.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._crowd = None
        self._ppo = None
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

        self._crowd = position_crowding(closes, oi, volumes, self.crowd_period)
        self._ppo, _, _ = ppo(closes, fast=12, slow=26, signal=9)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        crowd_val = self._crowd[i]
        ppo_val = self._ppo[i]
        atr_val = self._atr[i]
        if np.isnan(crowd_val) or np.isnan(ppo_val) or np.isnan(atr_val):
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
        if side == 0 and crowd_val < self.crowd_threshold and ppo_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and ppo_val < -1.0:
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
