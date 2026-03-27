import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.momentum.coppock import coppock
from indicators.trend.trend_intensity import trend_intensity
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV29(TimeSeriesStrategy):
    """
    策略简介：Coppock Curve反转信号 + Trend Intensity确认的做多策略（10min频率）。

    使用指标：
    - Coppock Curve(10,14,11): 长期动量曲线，从负转正为买入
    - Trend Intensity(14): 趋势强度 > 50 确认趋势中
    - ATR(14): 止损距离计算

    进场条件（做多）：Coppock > 0 且 从负转正（上穿零线） 且 TI > 50
    出场条件：ATR追踪止损 / 分层止盈 / Coppock转负

    优点：Coppock经典底部信号，TI过滤虚假反转
    缺点：Coppock信号少，可能错过中间段趋势
    """
    name = "mt_v29"
    warmup = 1000
    freq = "10min"

    coppock_wma: int = 10
    coppock_roc_long: int = 14
    coppock_roc_short: int = 11
    ti_period: int = 14
    ti_threshold: float = 50.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._coppock = None
        self._ti = None
        self._atr = None
        self._avg_volume = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._coppock = coppock(closes, wma=self.coppock_wma,
                                 roc_long=self.coppock_roc_long, roc_short=self.coppock_roc_short)
        self._ti = trend_intensity(closes, period=self.ti_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        cop = self._coppock[i]
        ti_val = self._ti[i]
        if np.isnan(atr_val) or np.isnan(cop) or np.isnan(ti_val):
            return
        if i < 1 or np.isnan(self._coppock[i - 1]):
            return

        self.bars_since_last_scale += 1
        prev_cop = self._coppock[i - 1]
        cop_cross_up = prev_cop < 0 and cop >= 0

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

        if side == 1 and self.entry_price > 0:
            profit_atr = (price - self.entry_price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_long(lots=max(1, lots // 3))
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_long(lots=max(1, lots // 3))
                self._took_profit_3atr = True
                return

        if side == 1 and cop < 0:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and cop_cross_up and ti_val > self.ti_threshold:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, cop, ti_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, cop, ti_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if cop < 0 or ti_val < self.ti_threshold:
            return False
        return True

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_stop_mult * atr_val * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = context.close_raw * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        return max(1, min(risk_lots, int(context.equity * 0.30 / margin)))

    def _reset_state(self):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
