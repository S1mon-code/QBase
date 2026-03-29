import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.momentum.tsi import tsi
from indicators.volume.volume_momentum import volume_momentum
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV261(TimeSeriesStrategy):
    """
    策略简介：TSI动量+量能动量跨类别组合，量价共振做多。
    使用指标：TSI(25,13,7) + Volume Momentum(14) + ATR
    进场条件：TSI上穿信号线且量能动量>1.5
    出场条件：ATR追踪止损 / TSI下穿信号线
    优点：动量+量价双确认，高质量信号
    缺点：4h频率信号频次较低
    """
    name = "mt_v261"
    warmup = 200
    freq = "4h"

    tsi_long: int = 25
    tsi_short: int = 13
    vol_mom_period: int = 14
    atr_trail_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._tsi_line = None
        self._tsi_signal = None
        self._vol_mom = None
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
        self._tsi_line, self._tsi_signal = tsi(closes, long=self.tsi_long, short=self.tsi_short, signal=7)
        self._vol_mom = volume_momentum(volumes, self.vol_mom_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        tsi_val = self._tsi_line[i]
        tsi_sig = self._tsi_signal[i]
        vm_val = self._vol_mom[i]
        if np.isnan(atr_val) or np.isnan(tsi_val) or np.isnan(tsi_sig) or np.isnan(vm_val) or atr_val <= 0:
            return
        if i < 1:
            return
        tsi_prev = self._tsi_line[i - 1]
        sig_prev = self._tsi_signal[i - 1]
        if np.isnan(tsi_prev) or np.isnan(sig_prev):
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and tsi_prev < sig_prev and tsi_val >= tsi_sig and vm_val > 1.5:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and tsi_val < tsi_sig:
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
