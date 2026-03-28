"""Strong Trend v136 — Speculation Index + HMA trend filter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.speculation_index import speculation_index
from indicators.trend.hma import hma
from indicators.volatility.atr import atr


class StrongTrendV136(TimeSeriesStrategy):
    """
    策略简介：Speculation Index 投机指数 + HMA 趋势方向确认的趋势策略。

    使用指标：
    - Speculation Index(20): 投机指数(成交量/持仓量)，高值表示投机活跃
    - HMA(20): Hull 移动平均线，价格在HMA上方确认多头
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - Speculation Index > spec_threshold（投机活跃，市场有方向）
    - 价格 > HMA（多头趋势确认）

    出场条件：
    - ATR 追踪止损触发
    - 价格 < HMA 时趋势反转信号退出

    优点：投机活跃度配合趋势方向，捕捉主力资金驱动的行情
    缺点：投机指数过高可能是行情末端信号
    """
    name = "strong_trend_v136"
    warmup = 60
    freq = "daily"

    spec_period: int = 20
    hma_period: int = 20
    spec_threshold: float = 1.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._spec_idx = None
        self._hma = None
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

        self._spec_idx = speculation_index(volumes, oi, self.spec_period)
        self._hma = hma(closes, self.hma_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        si = self._spec_idx[i]
        hma_val = self._hma[i]
        atr_val = self._atr[i]
        if np.isnan(si) or np.isnan(hma_val) or np.isnan(atr_val):
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
        if side == 0 and si > self.spec_threshold and price > hma_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and price < hma_val:
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
