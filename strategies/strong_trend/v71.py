"""Strong Trend v71 — OI Bollinger + ROC momentum."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.oi_bollinger import oi_bollinger
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr


class StrongTrendV71(TimeSeriesStrategy):
    """
    策略简介：OI Bollinger 持仓量布林 + ROC 价格动量策略。

    使用指标：
    - OI Bollinger(20, 2.0): OI 的布林通道，OI 突破上轨 = 异常建仓
    - ROC(14): 价格变化率，>roc_thresh 确认价格跟随
    - ATR(14): 追踪止损

    进场条件（做多）：
    - OI 突破 Bollinger 上轨（大量新仓涌入）
    - ROC > roc_threshold（价格动量正向，排除空头建仓）

    出场条件：
    - ATR 追踪止损
    - ROC < 0 且 OI 回到 Bollinger 中轨内（动量消退 + 仓位正常化）

    优点：OI Bollinger 量化异常建仓事件，与价格动量交叉验证
    缺点：OI 布林突破在流动性低的品种可能不准确
    """
    name = "strong_trend_v71"
    warmup = 60
    freq = "daily"

    oi_bb_period: int = 20
    oi_bb_std: float = 2.0
    roc_period: int = 14
    roc_threshold: float = 2.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._oi_bb = None  # returns tuple or single depending on impl
        self._roc = None
        self._atr = None
        self._oi = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        oi = context.get_full_oi_array()

        self._oi = oi
        self._oi_bb = oi_bollinger(oi, period=self.oi_bb_period, num_std=self.oi_bb_std)
        self._roc = rate_of_change(closes, period=self.roc_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        roc_val = self._roc[i]
        atr_val = self._atr[i]
        oi_val = self._oi[i]
        # oi_bollinger returns a signal array (positive = above upper band)
        oi_bb_val = self._oi_bb[i]
        if np.isnan(roc_val) or np.isnan(atr_val) or np.isnan(oi_bb_val):
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

        # Entry: OI breakout above Bollinger + positive price momentum
        if side == 0 and oi_bb_val > 0 and roc_val > self.roc_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: momentum gone + OI normalized
        elif side == 1 and roc_val < 0 and oi_bb_val <= 0:
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
