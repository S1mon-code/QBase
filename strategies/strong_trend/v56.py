"""Strong Trend v56 — Volume Weighted MACD + ATR breakout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.volume_weighted_macd import vwmacd
from indicators.volatility.atr import atr


class StrongTrendV56(TimeSeriesStrategy):
    """
    策略简介：Volume Weighted MACD 量价加权动量策略。

    使用指标：
    - VWMACD(12, 26, 9): 成交量加权MACD，比传统MACD更重视放量阶段
    - ATR(14): 追踪止损 + 突破确认

    进场条件（做多）：
    - VWMACD histogram > 0（量价动量向上）
    - VWMACD line > signal（金叉确认）
    - 价格突破前一日高点 + 0.5*ATR（动量突破）

    出场条件：
    - ATR 追踪止损
    - VWMACD histogram 转负退出

    优点：VWMACD 天然融合量价信息，单一指标蕴含丰富信号
    缺点：MACD 类指标存在固有滞后
    """
    name = "strong_trend_v56"
    warmup = 60
    freq = "daily"

    vwmacd_fast: int = 12
    vwmacd_slow: int = 26
    breakout_atr_mult: float = 0.5
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._vwmacd_line = None
        self._vwmacd_signal = None
        self._vwmacd_hist = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest = 0.0
        self.stop_price = 0.0
        self.prev_high = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        line, sig, hist = vwmacd(closes, volumes,
                                 fast=self.vwmacd_fast, slow=self.vwmacd_slow, signal=9)
        self._vwmacd_line = line
        self._vwmacd_signal = sig
        self._vwmacd_hist = hist
        self._atr = atr(highs, lows, closes, period=14)
        self._highs = highs

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        hist = self._vwmacd_hist[i]
        line = self._vwmacd_line[i]
        sig = self._vwmacd_signal[i]
        atr_val = self._atr[i]
        if np.isnan(hist) or np.isnan(atr_val) or i < 1:
            return

        prev_high = self._highs[i - 1]

        # Stop loss
        if side == 1:
            self.highest = max(self.highest, price)
            trailing = self.highest - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Entry: VWMACD bullish + price breakout
        if (side == 0 and hist > 0 and line > sig
                and price > prev_high + self.breakout_atr_mult * atr_val):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: histogram turns negative
        elif side == 1 and hist < 0:
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
