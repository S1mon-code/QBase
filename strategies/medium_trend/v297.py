import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.microstructure.adverse_selection import adverse_selection
from indicators.trend.psar import psar
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV297(TimeSeriesStrategy):
    """
    策略简介：逆向选择+PSAR方向确认，低逆向选择=更安全趋势跟踪。
    使用指标：Adverse Selection(20) + Parabolic SAR + ATR
    进场条件：逆向选择低（<中位数）且PSAR方向为多
    出场条件：ATR追踪止损 / PSAR翻空
    优点：过滤知情交易者主导的危险区间
    缺点：逆向选择指标解释复杂
    """
    name = "mt_v297"
    warmup = 300
    freq = "30min"

    as_period: int = 20
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._as = None
        self._psar_val = None
        self._psar_dir = None
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
        self._as = adverse_selection(closes, volumes, period=self.as_period)
        self._psar_val, self._psar_dir = psar(highs, lows)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        as_val = self._as[i]
        ps_dir = self._psar_dir[i]
        if np.isnan(atr_val) or np.isnan(as_val) or np.isnan(ps_dir) or atr_val <= 0:
            return
        if i < 1:
            return
        prev_dir = self._psar_dir[i - 1]
        if np.isnan(prev_dir):
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Low adverse selection + PSAR bullish
        if side == 0 and as_val < 0.5 and ps_dir == 1:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and ps_dir == -1:
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
