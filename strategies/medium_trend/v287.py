import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.seasonality.quarter_effect import quarter_end_effect
from indicators.momentum.kst import kst
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV287(TimeSeriesStrategy):
    """
    策略简介：季末效应+KST多周期动量确认。
    使用指标：Quarter End Effect(10) + KST + ATR
    进场条件：季末效应为正且KST上穿信号线
    出场条件：ATR追踪止损 / KST下穿信号线
    优点：季末资金流+多周期动量
    缺点：季末效应每年仅4次窗口
    """
    name = "mt_v287"
    warmup = 300
    freq = "1h"

    qtr_window: int = 10
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._qtr = None
        self._kst_line = None
        self._kst_signal = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        datetimes = context.get_full_datetime_array()
        self._qtr = quarter_end_effect(closes, datetimes, window=self.qtr_window)
        self._kst_line, self._kst_signal = kst(closes)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        qtr_val = self._qtr[i]
        kst_val = self._kst_line[i]
        kst_sig = self._kst_signal[i]
        if np.isnan(atr_val) or np.isnan(qtr_val) or np.isnan(kst_val) or np.isnan(kst_sig) or atr_val <= 0:
            return
        if i < 1:
            return
        kst_prev = self._kst_line[i - 1]
        sig_prev = self._kst_signal[i - 1]
        if np.isnan(kst_prev) or np.isnan(sig_prev):
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and qtr_val > 0 and kst_prev < sig_prev and kst_val >= kst_sig:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and kst_val < kst_sig:
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
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
