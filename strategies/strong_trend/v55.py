"""Strong Trend v55 — Klinger Oscillator + OI Divergence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.klinger import klinger
from indicators.volume.oi_divergence import oi_divergence
from indicators.volatility.atr import atr


class StrongTrendV55(TimeSeriesStrategy):
    """
    策略简介：Klinger Oscillator + OI Divergence 量价背离检测策略。

    使用指标：
    - Klinger Oscillator(34, 55, 13): 量价振荡，KVO > signal 看多
    - OI Divergence(20): OI-价格背离，>0 说明OI确认价格方向
    - ATR(14): 追踪止损

    进场条件（做多）：
    - Klinger KVO > Signal（多头量能占优）
    - OI Divergence > 0（持仓量确认价格方向，非背离）

    出场条件：
    - ATR 追踪止损
    - KVO 跌破 Signal 线退出

    优点：Klinger 将量能方向量化，OI背离过滤虚假信号
    缺点：双重确认条件严格，信号稀疏
    """
    name = "strong_trend_v55"
    warmup = 60
    freq = "daily"

    klinger_fast: int = 34
    klinger_slow: int = 55
    oi_div_period: int = 20
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._kvo = None
        self._kvo_signal = None
        self._oi_div = None
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

        kvo, kvo_sig = klinger(highs, lows, closes, volumes,
                               fast=self.klinger_fast, slow=self.klinger_slow, signal=13)
        self._kvo = kvo
        self._kvo_signal = kvo_sig
        self._oi_div = oi_divergence(closes, oi, period=self.oi_div_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        kvo_val = self._kvo[i]
        kvo_sig = self._kvo_signal[i]
        oi_div = self._oi_div[i]
        atr_val = self._atr[i]
        if np.isnan(kvo_val) or np.isnan(kvo_sig) or np.isnan(oi_div) or np.isnan(atr_val):
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

        # Entry: Klinger bullish crossover + OI confirmation
        if side == 0 and kvo_val > kvo_sig and oi_div > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: Klinger bearish
        elif side == 1 and kvo_val < kvo_sig:
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
