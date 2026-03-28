"""Strong Trend v60 — OI Accumulation + Price Momentum (ROC)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.oi_accumulation import oi_accumulation
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr


class StrongTrendV60(TimeSeriesStrategy):
    """
    策略简介：OI Accumulation 持仓累积 + ROC 价格动量策略。

    使用指标：
    - OI Accumulation(20): 持仓量累积分析，>0 表示多头建仓
    - ROC(20): 价格变化率，>roc_threshold 确认价格动量
    - ATR(14): 追踪止损

    进场条件（做多）：
    - OI Accumulation > 0（多头持续建仓）
    - ROC > roc_threshold（价格动量正向且显著）

    出场条件：
    - ATR 追踪止损
    - OI Accumulation < 0 且 ROC < 0（双重翻空）

    优点：OI累积直接反映市场参与者的持仓变化方向
    缺点：OI数据在某些合约中可能不够及时
    """
    name = "strong_trend_v60"
    warmup = 60
    freq = "daily"

    oi_acc_period: int = 20
    roc_period: int = 20
    roc_threshold: float = 3.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._oi_acc = None
        self._roc = None
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

        self._oi_acc = oi_accumulation(closes, oi, period=self.oi_acc_period)
        self._roc = rate_of_change(closes, period=self.roc_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        oi_acc = self._oi_acc[i]
        roc_val = self._roc[i]
        atr_val = self._atr[i]
        if np.isnan(oi_acc) or np.isnan(roc_val) or np.isnan(atr_val):
            return

        # Stop loss
        if side == 1:
            self.highest = max(self.highest, price)
            trailing = self.highest - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Entry: OI accumulation + price momentum
        if side == 0 and oi_acc > 0 and roc_val > self.roc_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: both bearish
        elif side == 1 and oi_acc < 0 and roc_val < 0:
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
