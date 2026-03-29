"""Strong Trend v133 — Smart Money Index + TEMA trend filter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.smart_money import smart_money_index
from indicators.trend.tema import tema
from indicators.volatility.atr import atr


class StrongTrendV133(TimeSeriesStrategy):
    """
    策略简介：Smart Money Index 聪明钱指数 + TEMA 趋势确认的趋势策略。

    使用指标：
    - Smart Money Index(20): 聪明钱流向，上升表示机构资金看多
    - TEMA(20): 三重指数平滑均线，价格在TEMA上方确认多头
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - Smart Money Index 上升（当前值 > 前一日值）
    - 价格 > TEMA（多头趋势确认）

    出场条件：
    - ATR 追踪止损触发
    - 价格 < TEMA 时趋势反转信号退出

    优点：跟踪聪明钱行为，TEMA 响应快减少滞后
    缺点：Smart Money Index 在日内数据不完整时可能失真
    """
    name = "strong_trend_v133"
    warmup = 60
    freq = "daily"

    smi_period: int = 20
    tema_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._smi = None
        self._tema = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        opens = context.get_full_open_array()
        volumes = context.get_full_volume_array()

        self._smi = smart_money_index(opens, closes, highs, lows, volumes, self.smi_period)
        self._tema = tema(closes, self.tema_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if i < 1:
            return

        smi_val = self._smi[i]
        smi_prev = self._smi[i - 1]
        tema_val = self._tema[i]
        atr_val = self._atr[i]
        if np.isnan(smi_val) or np.isnan(smi_prev) or np.isnan(tema_val) or np.isnan(atr_val):
            return

        smi_rising = smi_val > smi_prev

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
        if side == 0 and smi_rising and price > tema_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and price < tema_val:
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
