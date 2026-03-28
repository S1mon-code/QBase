"""
Strong Trend Strategy v101 — TTM Squeeze Release + Volume Momentum
===================================================================
Detects volatility compression via TTM Squeeze (Bollinger inside Keltner),
then enters on squeeze release with rising momentum and volume confirmation.

  1. TTM Squeeze    — detects low-volatility compression and release
  2. Volume Momentum — confirms institutional participation on breakout

Entry: squeeze was on (compressed), just released, momentum > 0 and rising,
       volume momentum confirms buying pressure.
Exit:  ATR trailing stop or momentum reversal.

LONG ONLY.

Usage:
    ./run.sh strategies/strong_trend/v101.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.ttm_squeeze import ttm_squeeze
from indicators.volume.volume_momentum import volume_momentum
from indicators.volatility.atr import atr


class StrongTrendV101(TimeSeriesStrategy):
    """
    策略简介：TTM Squeeze释放 + 成交量动量确认的波动率突破策略
    使用指标：TTM Squeeze（压缩检测）、Volume Momentum（量能确认）
    进场条件：Squeeze刚释放 + 动量为正且上升 + 成交量动量确认
    出场条件：ATR trailing stop 或动量反转
    优点：精准捕捉压缩后的爆发行情，量能过滤假突破
    缺点：横盘震荡期可能频繁触发假信号
    """
    name = "strong_trend_v101"
    warmup = 60
    freq = "daily"

    squeeze_bb: int = 20
    squeeze_kc_mult: float = 1.5
    vm_period: int = 14
    atr_trail_mult: float = 4.5
    mom_threshold: float = 0.0

    def __init__(self):
        super().__init__()
        self._squeeze = None
        self._momentum = None
        self._vm = None
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

        self._squeeze, self._momentum = ttm_squeeze(
            highs, lows, closes, bb_period=self.squeeze_bb, kc_mult=self.squeeze_kc_mult
        )
        self._vm = volume_momentum(volumes, self.vm_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        if i < 2:
            return

        cur_squeeze = self._squeeze[i]
        prev_squeeze = self._squeeze[i - 1]
        cur_mom = self._momentum[i]
        prev_mom = self._momentum[i - 1]
        vm_val = self._vm[i]
        atr_val = self._atr[i]

        if np.isnan(cur_mom) or np.isnan(atr_val) or np.isnan(vm_val):
            return

        # Squeeze just released: was on, now off
        squeeze_released = prev_squeeze and not cur_squeeze

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
        if side == 0:
            mom_rising = cur_mom > prev_mom and cur_mom > self.mom_threshold
            vol_confirm = vm_val > 1.0
            if squeeze_released and mom_rising and vol_confirm:
                lot_size = self._calc_lots(context, price, atr_val)
                if lot_size > 0:
                    context.buy(lot_size)
                    self.entry_price = price
                    self.stop_price = price - self.atr_trail_mult * atr_val
                    self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1:
            if cur_mom < 0 and not np.isnan(prev_mom) and prev_mom >= 0:
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
