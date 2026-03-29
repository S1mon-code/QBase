"""Strong Trend v63 — Volume Force + HMA (Hull MA) trend."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.volume_force import volume_force
from indicators.trend.hma import hma
from indicators.volatility.atr import atr


class StrongTrendV63(TimeSeriesStrategy):
    """
    策略简介：Volume Force 量力 + HMA 低延迟趋势策略。

    使用指标：
    - Volume Force(13): 量化成交量推动力，>0 看多
    - HMA(20): Hull MA 低延迟均线，斜率 > 0 确认上升
    - ATR(14): 追踪止损

    进场条件（做多）：
    - Volume Force > 0（量能推动力正向）
    - 价格 > HMA 且 HMA 上升（趋势方向向上）

    出场条件：
    - ATR 追踪止损
    - HMA 斜率转负（趋势动能衰减）

    优点：HMA 延迟极低，Volume Force 直接量化量能方向
    缺点：HMA 灵敏度高，可能在微调整中产生退出信号
    """
    name = "strong_trend_v63"
    warmup = 60
    freq = "daily"

    vf_period: int = 13
    hma_period: int = 20
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._vf = None
        self._hma = None
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

        self._vf = volume_force(closes, volumes, period=self.vf_period)
        self._hma = hma(closes, self.hma_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        vf_val = self._vf[i]
        hma_val = self._hma[i]
        atr_val = self._atr[i]
        if np.isnan(vf_val) or np.isnan(hma_val) or np.isnan(atr_val) or i < 2:
            return

        hma_prev = self._hma[i - 1]
        hma_slope_up = hma_val > hma_prev

        # Stop loss
        if side == 1:
            self.highest = max(self.highest, price)
            trailing = self.highest - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Entry: volume force bullish + price above rising HMA
        if side == 0 and vf_val > 0 and price > hma_val and hma_slope_up:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: HMA slope turns down
        elif side == 1 and not hma_slope_up:
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
