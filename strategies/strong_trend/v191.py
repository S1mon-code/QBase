"""
QBase Strong Trend Strategy v191 — Coppock Curve + Force Index
================================================================

策略简介：Coppock Curve是经典长期动量指标，原用于月度数据检测市场底部。
         当Coppock从负转正（底部反转）且Force Index确认买盘力量时做多。

使用指标：
  - Coppock Curve (wma=10, roc_long=14, roc_short=11): 长期动量
  - Force Index (period=13): 买卖力量
  - ATR (period=14): trailing stop

进场条件：
  1. Coppock > 0 且前一bar <= 0（零线上穿，底部反转）
  2. Force Index > 0（买盘主导）

出场条件：
  1. ATR trailing stop (mult=4.5)
  2. Coppock由正转负（动量丧失）

优点：Coppock底部信号可靠，Force Index量价确认
缺点：信号稀疏，长期指标延迟大
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.momentum.coppock import coppock
from indicators.volume.force_index import force_index
from indicators.volatility.atr import atr


class StrongTrendV191(TimeSeriesStrategy):
    """Coppock曲线底部反转 + Force Index策略。"""
    name = "strong_trend_v191"
    warmup = 60
    freq = "daily"

    coppock_wma: int = 10
    fi_period: int = 13
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._coppock = None
        self._fi = None
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

        self._coppock = coppock(closes, wma=self.coppock_wma, roc_long=14, roc_short=11)
        self._fi = force_index(closes, volumes, period=self.fi_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if i < 1:
            return

        cop = self._coppock[i]
        cop_prev = self._coppock[i - 1]
        fi_val = self._fi[i]
        atr_val = self._atr[i]
        if np.isnan(cop) or np.isnan(cop_prev) or np.isnan(fi_val) or np.isnan(atr_val):
            return

        # === Stop Loss Check ===
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # === Entry ===
        if side == 0 and cop > 0 and cop_prev <= 0 and fi_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and cop < 0 and cop_prev >= 0:
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
        max_lots = int(context.equity * 0.30 / margin)
        return max(1, min(risk_lots, max_lots))

    def _reset(self):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
