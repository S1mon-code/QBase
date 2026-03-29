"""Strong Trend v135 — Squeeze Detector + Keltner Channel breakout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.squeeze_detector import squeeze_probability
from indicators.trend.keltner import keltner
from indicators.volatility.atr import atr


class StrongTrendV135(TimeSeriesStrategy):
    """
    策略简介：Squeeze Detector 挤压概率 + Keltner 通道突破的趋势策略。

    使用指标：
    - Squeeze Probability(20): 挤压概率，高值表示即将突破
    - Keltner Channel(20, 10, 1.5): 价格突破上轨确认方向
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - Squeeze Probability > squeeze_threshold（挤压概率高）
    - 价格 > Keltner 上轨（突破确认）

    出场条件：
    - ATR 追踪止损触发
    - 价格 < Keltner 中轨时信号退出

    优点：挤压后突破是高概率趋势启动信号
    缺点：假突破时挤压解除可能导致快速止损
    """
    name = "strong_trend_v135"
    warmup = 60
    freq = "daily"

    squeeze_period: int = 20
    squeeze_threshold: float = 0.6
    kc_mult: float = 1.5
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._squeeze = None
        self._kc_upper = None
        self._kc_mid = None
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

        self._squeeze = squeeze_probability(closes, oi, volumes, self.squeeze_period)
        self._kc_upper, self._kc_mid, _ = keltner(highs, lows, closes, ema=20, atr=10, mult=self.kc_mult)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        sq = self._squeeze[i]
        kc_u = self._kc_upper[i]
        kc_m = self._kc_mid[i]
        atr_val = self._atr[i]
        if np.isnan(sq) or np.isnan(kc_u) or np.isnan(kc_m) or np.isnan(atr_val):
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
        if side == 0 and sq > self.squeeze_threshold and price > kc_u:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and price < kc_m:
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
