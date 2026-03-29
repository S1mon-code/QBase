import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.ml.robust_zscore import robust_zscore
from indicators.trend.ema import ema
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV252(TimeSeriesStrategy):
    """
    策略简介：Robust Z-Score均值偏离检测+EMA趋势过滤的做多策略。
    使用指标：Robust Z-Score(60) + EMA(50) + ATR
    进场条件：Robust Z-Score从负转正且价格在EMA上方
    出场条件：ATR追踪止损 / Z-Score极端高位回落
    优点：抗异常值，中位数基准更稳健
    缺点：长周期Z-Score反应偏慢
    """
    name = "mt_v252"
    warmup = 120
    freq = "daily"

    zscore_period: int = 60
    ema_period: int = 50
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._zscore = None
        self._ema = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        self._zscore = robust_zscore(closes, period=self.zscore_period)
        self._ema = ema(closes, period=self.ema_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        z_val = self._zscore[i]
        ema_val = self._ema[i]
        if np.isnan(atr_val) or np.isnan(z_val) or np.isnan(ema_val) or atr_val <= 0:
            return
        if i < 1:
            return
        z_prev = self._zscore[i - 1]
        if np.isnan(z_prev):
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and z_prev < 0 and z_val >= 0 and price > ema_val:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and z_val > 2.5:
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
