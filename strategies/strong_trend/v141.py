"""Strong Trend v141 — OI Breakout + Bollinger Bands confirmation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.oi_breakout import oi_breakout
from indicators.volatility.bollinger import bollinger_bands
from indicators.volatility.atr import atr


class StrongTrendV141(TimeSeriesStrategy):
    """
    策略简介：OI Breakout 持仓量突破 + Bollinger Bands 价格突破双确认趋势策略。

    使用指标：
    - OI Breakout(20, 2.0): 持仓量突破检测，突破=资金大量涌入
    - Bollinger Bands(20, 2.0): 价格突破上轨确认方向
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - OI Breakout 触发（持仓量突破阈值）
    - 价格 > Bollinger 上轨（价格也在突破）

    出场条件：
    - ATR 追踪止损触发
    - 价格 < Bollinger 中轨时信号退出

    优点：OI+价格双重突破确认，信号可靠性高
    缺点：双重突破条件严格，入场机会稀少
    """
    name = "strong_trend_v141"
    warmup = 60
    freq = "daily"

    oi_break_period: int = 20
    oi_break_threshold: float = 2.0
    bb_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._oi_break = None
        self._bb_upper = None
        self._bb_mid = None
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

        self._oi_break = oi_breakout(oi, self.oi_break_period, self.oi_break_threshold)
        self._bb_upper, self._bb_mid, _ = bollinger_bands(closes, self.bb_period, std=2.0)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        ob = self._oi_break[i]
        bb_u = self._bb_upper[i]
        bb_m = self._bb_mid[i]
        atr_val = self._atr[i]
        if np.isnan(ob) or np.isnan(bb_u) or np.isnan(bb_m) or np.isnan(atr_val):
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
        if side == 0 and ob > 0 and price > bb_u:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and price < bb_m:
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
