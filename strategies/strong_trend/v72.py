"""Strong Trend v72 — OI Relative Strength + ADX trend filter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.oi_relative_strength import oi_relative_strength
from indicators.trend.adx import adx
from indicators.volatility.atr import atr


class StrongTrendV72(TimeSeriesStrategy):
    """
    策略简介：OI Relative Strength 持仓相对强度 + ADX 趋势环境策略。

    使用指标：
    - OI Relative Strength(20): OI相对成交量的强度，高值=持仓积累
    - ADX(14): 趋势强度，>adx_thresh 确认趋势环境
    - ATR(14): 追踪止损

    进场条件（做多）：
    - OI RS > oi_rs_threshold（持仓积累强于交易活动，长线资金入场）
    - ADX > adx_threshold（趋势环境确认）
    - 价格创20日新高（方向确认）

    出场条件：
    - ATR 追踪止损
    - ADX < 20（趋势环境消失）

    优点：OI RS 区分投机交易和长线建仓，ADX 过滤震荡
    缺点：ADX 回落慢，可能在趋势衰减初期不能及时退出
    """
    name = "strong_trend_v72"
    warmup = 60
    freq = "daily"

    oi_rs_period: int = 20
    oi_rs_threshold: float = 1.2
    adx_period: int = 14
    adx_threshold: float = 25.0
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._oi_rs = None
        self._adx = None
        self._atr = None
        self._highs = None

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

        self._oi_rs = oi_relative_strength(oi, volumes, period=self.oi_rs_period)
        self._adx = adx(highs, lows, closes, period=self.adx_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._highs = highs

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        oi_rs = self._oi_rs[i]
        adx_val = self._adx[i]
        atr_val = self._atr[i]
        if np.isnan(oi_rs) or np.isnan(adx_val) or np.isnan(atr_val) or i < 20:
            return

        # 20-day high
        high_20 = np.nanmax(self._highs[max(0, i-19):i+1])

        # Stop loss
        if side == 1:
            self.highest = max(self.highest, price)
            trailing = self.highest - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Entry: OI accumulation + trending + breakout
        if (side == 0 and oi_rs > self.oi_rs_threshold
                and adx_val > self.adx_threshold
                and price >= high_20):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: trend environment gone
        elif side == 1 and adx_val < 20:
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
