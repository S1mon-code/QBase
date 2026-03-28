"""Strong Trend v64 — OI Climax + Chandelier Exit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.oi_climax import oi_climax
from indicators.volatility.chandelier_exit import chandelier_exit
from indicators.volatility.atr import atr


class StrongTrendV64(TimeSeriesStrategy):
    """
    策略简介：OI Climax 持仓高潮检测 + Chandelier Exit 动态止损策略。

    使用指标：
    - OI Climax(20, 2.0): 检测OI极端放大事件，climax > 0 为高潮信号
    - Chandelier Exit(22, 3.0): 基于最高价的动态止损线
    - ATR(14): 仓位计算

    进场条件（做多）：
    - OI Climax 触发（持仓量爆发式增长，大资金入场）
    - 价格 > Chandelier Long Line（确认在上升趋势中）

    出场条件：
    - Chandelier Exit 止损触发
    - ATR 追踪止损（取两者较高者）

    优点：OI Climax 捕捉大资金入场时机，高精度低频信号
    缺点：Climax 事件稀少，持仓等待时间长
    """
    name = "strong_trend_v64"
    warmup = 60
    freq = "daily"

    oi_climax_period: int = 20
    oi_climax_thresh: float = 2.0
    chand_period: int = 22
    chand_mult: float = 3.0
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._oi_climax = None
        self._chand_long = None
        self._chand_short = None
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

        self._oi_climax = oi_climax(oi, volumes, period=self.oi_climax_period,
                                     threshold=self.oi_climax_thresh)
        chand_long, chand_short = chandelier_exit(highs, lows, closes,
                                                    period=self.chand_period,
                                                    mult=self.chand_mult)
        self._chand_long = chand_long
        self._chand_short = chand_short
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        climax = self._oi_climax[i]
        ch_long = self._chand_long[i]
        atr_val = self._atr[i]
        if np.isnan(climax) or np.isnan(ch_long) or np.isnan(atr_val):
            return

        # Stop loss: higher of chandelier and ATR trail
        if side == 1:
            self.highest = max(self.highest, price)
            atr_trail = self.highest - self.atr_trail_mult * atr_val
            combined_stop = max(ch_long, atr_trail)
            self.stop_price = max(self.stop_price, combined_stop)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Entry: OI climax event + price above chandelier (uptrend)
        if side == 0 and climax > 0 and price > ch_long:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

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
