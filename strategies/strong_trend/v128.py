"""Strong Trend v128 — Volume Imbalance + ADX trend strength."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.microstructure.volume_imbalance import volume_imbalance
from indicators.trend.adx import adx
from indicators.volatility.atr import atr


class StrongTrendV128(TimeSeriesStrategy):
    """
    策略简介：Volume Imbalance 买卖量失衡 + ADX 趋势强度确认的趋势策略。

    使用指标：
    - Volume Imbalance(20): 买卖量不平衡度，正值表示买方主导
    - ADX(14): 趋势强度，>25 确认强趋势
    - ATR(14): 追踪止损距离计算

    进场条件（做多）：
    - Volume Imbalance > imb_threshold（买方主导）
    - ADX > adx_threshold（强趋势环境）

    出场条件：
    - ATR 追踪止损触发
    - ADX < 20 时趋势减弱信号退出

    优点：量能失衡确认方向，ADX 过滤震荡行情
    缺点：震荡市ADX长期低位，入场机会少
    """
    name = "strong_trend_v128"
    warmup = 60
    freq = "daily"

    imb_period: int = 20
    adx_period: int = 14
    imb_threshold: float = 0.3
    adx_threshold: float = 25.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._imb = None
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

        self._imb = volume_imbalance(closes, volumes, self.imb_period)
        self._adx = adx(highs, lows, closes, self.adx_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        imb_val = self._imb[i]
        adx_val = self._adx[i]
        atr_val = self._atr[i]
        if np.isnan(imb_val) or np.isnan(adx_val) or np.isnan(atr_val):
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
        if side == 0 and imb_val > self.imb_threshold and adx_val > self.adx_threshold:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit
        elif side == 1 and adx_val < 20.0:
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
