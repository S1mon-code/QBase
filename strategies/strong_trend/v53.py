"""Strong Trend v53 — OBV + Volume Spike accumulation breakout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.obv import obv
from indicators.volume.volume_spike import volume_spike
from indicators.volatility.atr import atr
from indicators.trend.ema import ema


class StrongTrendV53(TimeSeriesStrategy):
    """
    策略简介：OBV 累积趋势 + Volume Spike 突破确认策略。

    使用指标：
    - OBV: 累积成交量，OBV 上升说明持续吸筹
    - OBV EMA(20): OBV 趋势方向
    - Volume Spike(20, 2.0): 成交量突破检测
    - ATR(14): 追踪止损

    进场条件（做多）：
    - OBV > OBV EMA（累积量趋势向上）
    - Volume Spike 触发（放量突破）

    出场条件：
    - ATR 追踪止损
    - OBV 跌破 OBV EMA 且无 spike 时退出

    优点：OBV 追踪长期资金流向，Spike 确认突破时机
    缺点：OBV 对缺口敏感，可能在高波动期产生误判
    """
    name = "strong_trend_v53"
    warmup = 60
    freq = "daily"

    obv_ema_period: int = 20
    spike_threshold: float = 2.0
    atr_trail_mult: float = 4.5

    def __init__(self):
        super().__init__()
        self._obv = None
        self._obv_ema = None
        self._spike = None
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

        self._obv = obv(closes, volumes)
        self._obv_ema = ema(self._obv, self.obv_ema_period)
        self._spike = volume_spike(volumes, period=20, threshold=self.spike_threshold)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        obv_val = self._obv[i]
        obv_ema_val = self._obv_ema[i]
        spike_val = self._spike[i]
        atr_val = self._atr[i]
        if np.isnan(obv_val) or np.isnan(obv_ema_val) or np.isnan(atr_val):
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

        # Entry: OBV trending up + volume spike
        if side == 0 and obv_val > obv_ema_val and spike_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: OBV trend broken
        elif side == 1 and obv_val < obv_ema_val:
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
