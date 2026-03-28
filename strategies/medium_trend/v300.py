import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.microstructure.trade_clustering import trade_clustering
from indicators.volume.volume_efficiency import volume_efficiency
from indicators.volatility.atr import atr

_SPEC_MANAGER = ContractSpecManager()


class MediumTrendV300(TimeSeriesStrategy):
    """
    策略简介：交易聚集度+成交量效率双重微观结构策略。
    使用指标：Trade Clustering(20) + Volume Efficiency(20) + ATR
    进场条件：交易聚集度高（集中交易期）且成交量效率为正
    出场条件：ATR追踪止损 / 成交量效率转负
    优点：聚集交易+高效价格移动=强趋势信号
    缺点：微观结构指标在30min可能过于敏感
    """
    name = "mt_v300"
    warmup = 300
    freq = "30min"

    tc_period: int = 20
    ve_period: int = 20
    atr_trail_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._tc = None
        self._ve = None
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
        self._tc = trade_clustering(volumes, period=self.tc_period)
        self._ve = volume_efficiency(closes, volumes, period=self.ve_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        atr_val = self._atr[i]
        tc_val = self._tc[i]
        ve_val = self._ve[i]
        if np.isnan(atr_val) or np.isnan(tc_val) or np.isnan(ve_val) or atr_val <= 0:
            return

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        if side == 0 and tc_val > 0.5 and ve_val > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price
        elif side == 1 and ve_val < -0.3:
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
