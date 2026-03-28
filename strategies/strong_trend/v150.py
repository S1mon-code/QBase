"""Strong Trend v150 — Adverse Selection + Volume Momentum confirmation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.microstructure.adverse_selection import adverse_selection
from indicators.volume.volume_momentum import volume_momentum
from indicators.volatility.atr import atr


class StrongTrendV150(TimeSeriesStrategy):
    """
    策略简介：Adverse Selection 逆向选择成本低 + Volume Momentum 放量确认的趋势策略。

    使用指标：
    - Adverse Selection(20): 逆向选择成本，低值表示信息不对称低，适合入场
    - Volume Momentum(14): 成交量动量，>2.0 确认放量
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - Adverse Selection < as_threshold（逆向选择成本低，安全入场）
    - Volume Momentum > vol_threshold（放量确认）

    出场条件：
    - ATR 追踪止损触发
    - Volume Momentum < 0.5 时量能衰退信号退出

    优点：低逆向选择+放量组合确保入场时信息环境良好
    缺点：逆向选择度量基于间接估算，精度有限
    """
    name = "strong_trend_v150"
    warmup = 60
    freq = "daily"

    as_period: int = 20
    vol_mom_period: int = 14
    as_threshold: float = 0.5
    vol_threshold: float = 2.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._as = None
        self._vol_mom = None
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

        self._as = adverse_selection(closes, volumes, self.as_period)
        self._vol_mom = volume_momentum(volumes, self.vol_mom_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        as_val = self._as[i]
        vm = self._vol_mom[i]
        atr_val = self._atr[i]
        if np.isnan(as_val) or np.isnan(vm) or np.isnan(atr_val):
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
        if side == 0 and as_val < self.as_threshold and vm > self.vol_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and vm < 0.5:
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
