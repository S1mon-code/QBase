"""Strong Trend v57 — Twiggs Money Flow + Supertrend."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.twiggs import twiggs_money_flow
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr


class StrongTrendV57(TimeSeriesStrategy):
    """
    策略简介：Twiggs Money Flow 资金流 + Supertrend 趋势跟踪策略。

    使用指标：
    - Twiggs Money Flow(21): 改良资金流指标，>0 看多
    - Supertrend(10, 3.0): 趋势方向 + 动态止损线
    - ATR(14): 仓位计算

    进场条件（做多）：
    - Twiggs MF > tmf_threshold（持续资金流入）
    - Supertrend 方向 = 1（上升趋势）

    出场条件：
    - ATR 追踪止损（取 Supertrend 线和 ATR 止损的较高者）
    - Twiggs MF < 0 退出

    优点：Supertrend 提供动态止损，Twiggs MF 比 CMF 更平滑
    缺点：双指标确认在横盘期可能反复触发
    """
    name = "strong_trend_v57"
    warmup = 60
    freq = "daily"

    tmf_period: int = 21
    st_period: int = 10
    st_mult: float = 3.0
    tmf_threshold: float = 0.05
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._tmf = None
        self._st_line = None
        self._st_dir = None
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

        self._tmf = twiggs_money_flow(highs, lows, closes, volumes, period=self.tmf_period)
        st_line, st_dir = supertrend(highs, lows, closes,
                                      period=self.st_period, mult=self.st_mult)
        self._st_line = st_line
        self._st_dir = st_dir
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        tmf_val = self._tmf[i]
        st_dir = self._st_dir[i]
        st_line = self._st_line[i]
        atr_val = self._atr[i]
        if np.isnan(tmf_val) or np.isnan(st_dir) or np.isnan(atr_val):
            return

        # Stop loss: use higher of supertrend line and ATR trail
        if side == 1:
            self.highest = max(self.highest, price)
            atr_trail = self.highest - self.atr_trail_mult * atr_val
            combined_stop = max(st_line if st_dir == 1 else 0.0, atr_trail)
            self.stop_price = max(self.stop_price, combined_stop)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Entry: positive money flow + uptrend
        if side == 0 and tmf_val > self.tmf_threshold and st_dir == 1:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: money flow reversal
        elif side == 1 and tmf_val < 0:
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
