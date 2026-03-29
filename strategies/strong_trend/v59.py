"""Strong Trend v59 — Volume RSI + Aroon trend confirmation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.volume_weighted_rsi import volume_rsi
from indicators.trend.aroon import aroon
from indicators.volatility.atr import atr


class StrongTrendV59(TimeSeriesStrategy):
    """
    策略简介：Volume RSI 量价 RSI + Aroon 趋势确认策略。

    使用指标：
    - Volume RSI(14): 成交量加权 RSI，>60 看多
    - Aroon Up/Down(25): Aroon Up > 70 确认上升趋势
    - ATR(14): 追踪止损

    进场条件（做多）：
    - Volume RSI > vrsi_threshold（量价动量偏多）
    - Aroon Up > 70 且 Aroon Up > Aroon Down（趋势确认）

    出场条件：
    - ATR 追踪止损
    - Aroon Down > Aroon Up（趋势逆转）

    优点：Volume RSI 比普通 RSI 更贴近市场资金意愿
    缺点：Aroon 在窄幅震荡时可能误判趋势方向
    """
    name = "strong_trend_v59"
    warmup = 60
    freq = "daily"

    vrsi_period: int = 14
    aroon_period: int = 25
    vrsi_threshold: float = 60.0
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._vrsi = None
        self._aroon_up = None
        self._aroon_down = None
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

        self._vrsi = volume_rsi(closes, volumes, period=self.vrsi_period)
        up, down, osc = aroon(highs, lows, period=self.aroon_period)
        self._aroon_up = up
        self._aroon_down = down
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        vrsi_val = self._vrsi[i]
        ar_up = self._aroon_up[i]
        ar_down = self._aroon_down[i]
        atr_val = self._atr[i]
        if np.isnan(vrsi_val) or np.isnan(ar_up) or np.isnan(atr_val):
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

        # Entry: volume RSI bullish + Aroon uptrend
        if (side == 0 and vrsi_val > self.vrsi_threshold
                and ar_up > 70 and ar_up > ar_down):
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: Aroon bearish flip
        elif side == 1 and ar_down > ar_up:
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
