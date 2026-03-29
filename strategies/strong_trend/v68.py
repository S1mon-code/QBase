"""Strong Trend v68 — OI Adjusted Volume + Supertrend."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.oi_adjusted_volume import oi_adjusted_volume
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr


class StrongTrendV68(TimeSeriesStrategy):
    """
    策略简介：OI Adjusted Volume 持仓调整量 + Supertrend 趋势策略。

    使用指标：
    - OI Adjusted Volume(20): 用OI调整成交量，去除换仓噪音
    - Supertrend(10, 3.0): 趋势方向 + 动态支撑
    - ATR(14): 仓位计算

    进场条件（做多）：
    - OI Adjusted Volume 上升（过去3日均值 > 过去10日均值，真实参与度增加）
    - Supertrend 方向 = 1（上升趋势确认）

    出场条件：
    - Supertrend 翻转（方向变为 -1）
    - ATR 追踪止损

    优点：OI Adjusted Volume 剔除换仓虚量，信号更纯粹
    缺点：调整算法假设OI变化即为新仓，可能不完全准确
    """
    name = "strong_trend_v68"
    warmup = 60
    freq = "daily"

    oi_adj_period: int = 20
    st_period: int = 10
    st_mult: float = 3.0
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._oi_adj_vol = None
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
        oi = context.get_full_oi_array()

        self._oi_adj_vol = oi_adjusted_volume(volumes, oi, period=self.oi_adj_period)
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

        oi_adj = self._oi_adj_vol[i]
        st_dir = self._st_dir[i]
        st_line = self._st_line[i]
        atr_val = self._atr[i]
        if np.isnan(oi_adj) or np.isnan(st_dir) or np.isnan(atr_val) or i < 10:
            return

        # Short-term vs medium-term OI-adjusted volume
        short_avg = np.nanmean(self._oi_adj_vol[max(0, i-2):i+1])
        med_avg = np.nanmean(self._oi_adj_vol[max(0, i-9):i+1])

        # Stop loss
        if side == 1:
            self.highest = max(self.highest, price)
            atr_trail = self.highest - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, atr_trail)
            if price <= self.stop_price or st_dir == -1:
                context.close_long()
                self._reset()
                return

        # Entry: rising adjusted volume + supertrend up
        if side == 0 and short_avg > med_avg and st_dir == 1:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = max(st_line, price - self.atr_trail_mult * atr_val)

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
