"""Strong Trend v54 — CMF (Chaikin Money Flow) + ADX trend filter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.cmf import cmf
from indicators.trend.adx import adx
from indicators.volatility.atr import atr


class StrongTrendV54(TimeSeriesStrategy):
    """
    策略简介：CMF 资金流向 + ADX 趋势强度过滤策略。

    使用指标：
    - CMF(20): Chaikin Money Flow，>0.1 确认资金持续流入
    - ADX(14): 趋势强度，>25 确认趋势环境
    - ATR(14): 追踪止损

    进场条件（做多）：
    - CMF > cmf_threshold（资金流入显著）
    - ADX > adx_threshold（趋势环境确认）

    出场条件：
    - ATR 追踪止损
    - CMF < 0 时信号退出（资金流出）

    优点：CMF 直接度量资金流向，ADX 过滤震荡期
    缺点：ADX 滞后，可能错过趋势初期
    """
    name = "strong_trend_v54"
    warmup = 60
    freq = "daily"

    cmf_period: int = 20
    adx_period: int = 14
    cmf_threshold: float = 0.10
    adx_threshold: float = 25.0
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._cmf = None
        self._adx = None
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

        self._cmf = cmf(highs, lows, closes, volumes, period=self.cmf_period)
        self._adx = adx(highs, lows, closes, period=self.adx_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        cmf_val = self._cmf[i]
        adx_val = self._adx[i]
        atr_val = self._atr[i]
        if np.isnan(cmf_val) or np.isnan(adx_val) or np.isnan(atr_val):
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

        # Entry: strong money flow + trending market
        if side == 0 and cmf_val > self.cmf_threshold and adx_val > self.adx_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: money flow reversal
        elif side == 1 and cmf_val < 0:
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
