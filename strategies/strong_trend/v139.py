"""Strong Trend v139 — Commitment Ratio + EMA Ribbon trend."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.commitment_ratio import commitment_ratio
from indicators.trend.ema_ribbon import ema_ribbon_signal
from indicators.volatility.atr import atr


class StrongTrendV139(TimeSeriesStrategy):
    """
    策略简介：Commitment Ratio 承诺比率 + EMA Ribbon 多头排列确认的趋势策略。

    使用指标：
    - Commitment Ratio(20): OI/Volume 承诺比率，高值表示持仓意愿强
    - EMA Ribbon Signal: 多EMA排列信号，正值表示多头排列
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - Commitment Ratio > commit_threshold（持仓承诺强）
    - EMA Ribbon Signal > 0（多头排列确认）

    出场条件：
    - ATR 追踪止损触发
    - EMA Ribbon Signal < 0 时空头排列信号退出

    优点：承诺比率反映市场参与者的持仓决心
    缺点：EMA Ribbon 信号滞后较大，在趋势转折点反应慢
    """
    name = "strong_trend_v139"
    warmup = 60
    freq = "daily"

    commit_period: int = 20
    commit_threshold: float = 1.2
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._commit = None
        self._ribbon = None
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

        self._commit = commitment_ratio(oi, volumes, self.commit_period)
        self._ribbon = ema_ribbon_signal(closes, periods=[8, 13, 21, 34, 55])
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        cr = self._commit[i]
        rb = self._ribbon[i]
        atr_val = self._atr[i]
        if np.isnan(cr) or np.isnan(rb) or np.isnan(atr_val):
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
        if side == 0 and cr > self.commit_threshold and rb > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and rb < 0:
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
