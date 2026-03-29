"""Strong Trend v138 — Market Depth Proxy + Volume Spike confirmation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.market_depth_proxy import depth_proxy
from indicators.volume.volume_spike import volume_spike
from indicators.volatility.atr import atr


class StrongTrendV138(TimeSeriesStrategy):
    """
    策略简介：Market Depth Proxy 市场深度代理 + Volume Spike 放量确认的趋势策略。

    使用指标：
    - Market Depth Proxy(20): 市场深度估计，高值表示深度好
    - Volume Spike(20, 2.0): 成交量突增检测，确认资金涌入
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - Market Depth > depth_threshold（市场深度充足）
    - Volume Spike 在近3根K线内触发（放量确认）

    出场条件：
    - ATR 追踪止损触发
    - Market Depth < exit_depth 时深度不足信号退出

    优点：深度好+放量的组合确保入场时流动性充足
    缺点：深度代理指标基于间接估算，精度有限
    """
    name = "strong_trend_v138"
    warmup = 60
    freq = "daily"

    depth_period: int = 20
    depth_threshold: float = 1.5
    vol_spike_threshold: float = 2.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._depth = None
        self._vol_spike = None
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

        self._depth = depth_proxy(highs, lows, volumes, self.depth_period)
        self._vol_spike = volume_spike(volumes, period=20, threshold=self.vol_spike_threshold)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        dp = self._depth[i]
        atr_val = self._atr[i]
        if np.isnan(dp) or np.isnan(atr_val):
            return

        recent_spike = np.any(self._vol_spike[max(0, i - 2):i + 1])

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
        if side == 0 and dp > self.depth_threshold and recent_spike:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and dp < 0.5:
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
