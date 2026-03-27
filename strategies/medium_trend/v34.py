import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.trend.alma import alma
from indicators.momentum.rmi import rmi
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV34(TimeSeriesStrategy):
    """
    策略简介：ALMA趋势方向 + Relative Momentum Index确认的做多策略（10min频率）。

    使用指标：
    - ALMA(9,0.85,6): Arnaud Legoux均线，高斯加权低噪音
    - RMI(14,5): 相对动量指数，>50为多头动量
    - ATR(14): 止损距离计算

    进场条件（做多）：价格 > ALMA 且 ALMA斜率>0 且 RMI > 50
    出场条件：ATR追踪止损 / 分层止盈 / RMI < 40

    优点：ALMA高斯加权平滑且低延迟，RMI加入lookback消除噪音
    缺点：ALMA参数(offset/sigma)敏感
    """
    name = "mt_v34"
    warmup = 1000
    freq = "10min"

    alma_period: int = 9
    alma_offset: float = 0.85
    alma_sigma: float = 6.0
    rmi_period: int = 14
    rmi_lookback: int = 5
    rmi_entry: float = 50.0
    rmi_exit: float = 40.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._alma = None
        self._rmi = None
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

        self._alma = alma(closes, period=self.alma_period, offset=self.alma_offset, sigma=self.alma_sigma)
        self._rmi = rmi(closes, period=self.rmi_period, lookback=self.rmi_lookback)
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
        alma_val = self._alma[i]
        rmi_val = self._rmi[i]
        if np.isnan(atr_val) or np.isnan(alma_val) or np.isnan(rmi_val):
            return
        if i < 1 or np.isnan(self._alma[i - 1]):
            return

        self.bars_since_last_scale += 1
        alma_slope = alma_val - self._alma[i - 1]
        bullish = price > alma_val and alma_slope > 0 and rmi_val > self.rmi_entry

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

        if side == 1 and rmi_val < self.rmi_exit:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and bullish:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, bullish):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, bullish):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if not bullish:
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
