import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.structure.oi_velocity import oi_velocity
from indicators.volume.volume_momentum import volume_momentum
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV276(TimeSeriesStrategy):
    """
    策略简介：OI加速度+成交量动量双重确认，新资金+放量共振。
    使用指标：OI Velocity(5) + Volume Momentum(14) + ATR
    进场条件：OI加速为正且成交量动量>1.5
    出场条件：ATR追踪止损 / OI加速转负
    优点：短期OI变化率+量能共振，信号及时
    缺点：OI Velocity噪声较大
    """
    name = "mt_v276"
    warmup = 200
    freq = "1h"

    oi_vel_period: int = 5
    vol_mom_period: int = 14
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._oi_vel = None
        self._vol_mom = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()
        self._oi_vel = oi_velocity(oi, period=self.oi_vel_period)
        self._vol_mom = volume_momentum(volumes, self.vol_mom_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        oi_v = self._oi_vel[i]
        vm = self._vol_mom[i]
        if np.isnan(atr_val) or np.isnan(oi_v) or np.isnan(vm) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and oi_v > 0 and vm > 1.5:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and oi_v < -0.5:
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
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
