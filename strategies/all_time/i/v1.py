import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.i.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.volatility.atr import atr
from indicators.trend.supertrend import supertrend
from indicators.momentum.roc import rate_of_change
from indicators.volume.volume_spike import volume_spike

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV1(TimeSeriesStrategy):
    """
    策略简介：Supertrend方向确认 + ROC动量过滤 + 成交量异动放大的日线趋势策略。

    使用指标：
    - Supertrend(10, 3.0): 趋势方向判断，direction=1做多/-1做空
    - ROC(20): 动量方向确认，正值=上涨动量
    - Volume Spike(20, 2.0): 成交量异动检测，放量确认突破有效性

    进场条件（做多）：Supertrend方向=1 且 ROC>roc_threshold 且 出现放量
    进场条件（做空）：Supertrend方向=-1 且 ROC<-roc_threshold 且 出现放量

    出场条件：
    - ATR追踪止损（最高/最低价回撤N×ATR）
    - 分层止盈（3ATR减1/3，5ATR再减1/3）
    - Supertrend方向反转

    优点：三重确认（趋势+动量+量能），假突破过滤能力强
    缺点：放量要求可能错过低波动期的有效趋势
    """
    name = "i_alltime_v1"
    warmup = 250
    freq = "daily"

    st_period: int = 10
    st_mult: float = 3.0
    roc_period: int = 20
    roc_threshold: float = 2.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._st_dir = None
        self._roc = None
        self._vol_spike = None
        self._avg_volume = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.direction = 0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)
        _, self._st_dir = supertrend(highs, lows, closes, self.st_period, self.st_mult)
        self._roc = rate_of_change(closes, self.roc_period)
        self._vol_spike = volume_spike(volumes, 20, 2.0)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        st_dir = self._st_dir[i]
        roc_val = self._roc[i]
        spike = self._vol_spike[i]
        if np.isnan(st_dir) or np.isnan(roc_val) or np.isnan(spike):
            return

        self.bars_since_last_scale += 1

        # 1. Stop loss
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return
        elif side == -1:
            self.lowest_since_entry = min(self.lowest_since_entry, price)
            trailing = self.lowest_since_entry + self.atr_stop_mult * atr_val
            self.stop_price = min(self.stop_price, trailing)
            if price >= self.stop_price:
                context.close_short()
                self._reset_state()
                return

        # 2. Tiered profit-taking
        if side != 0 and self.entry_price > 0:
            profit_atr = ((price - self.entry_price) / atr_val) if side == 1 else ((self.entry_price - price) / atr_val)
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                cl = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=cl)
                else:
                    context.close_short(lots=cl)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                cl = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=cl)
                else:
                    context.close_short(lots=cl)
                self._took_profit_3atr = True
                return

        # 3. Signal-based exit: Supertrend reversal
        if side == 1 and st_dir != 1:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and st_dir != -1:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if st_dir == 1 and roc_val > self.roc_threshold and spike > 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
                    self.direction = 1
            elif st_dir == -1 and roc_val < -self.roc_threshold and spike > 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
                    self.direction = -1

        # 5. Scale-in
        elif side != 0 and self._should_add(price, atr_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if side == 1:
                    context.buy(add_lots)
                else:
                    context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1 and price < self.entry_price + atr_val:
            return False
        if self.direction == -1 and price > self.entry_price - atr_val:
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
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.direction = 0
