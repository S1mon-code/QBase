import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.trend.aroon import aroon
from indicators.volatility.historical_vol import historical_volatility
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV270(TimeSeriesStrategy):
    """
    策略简介：Aroon趋势检测+历史波动率过滤组合。
    使用指标：Aroon(25) + Historical Volatility(20) + ATR
    进场条件：Aroon Up>70且Aroon Osc>50且波动率在合理范围
    出场条件：ATR追踪止损 / Aroon Up<30
    优点：Aroon直观检测趋势强度+波动率过滤异常
    缺点：Aroon在横盘时信号模糊
    """
    name = "mt_v270"
    warmup = 200
    freq = "4h"

    aroon_period: int = 25
    vol_period: int = 20
    atr_trail_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._aroon_up = None
        self._aroon_down = None
        self._aroon_osc = None
        self._hvol = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(highs, lows, period=self.aroon_period)
        self._hvol = historical_volatility(closes, period=self.vol_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        ar_up = self._aroon_up[i]
        ar_osc = self._aroon_osc[i]
        hvol = self._hvol[i]
        if np.isnan(atr_val) or np.isnan(ar_up) or np.isnan(ar_osc) or np.isnan(hvol) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Filter: volatility between 10% and 80% annualized
        vol_ok = 0.10 < hvol < 0.80
        if side == 0 and ar_up > 70 and ar_osc > 50 and vol_ok:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and ar_up < 30:
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
