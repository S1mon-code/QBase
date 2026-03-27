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
from indicators.trend.adx import adx
from indicators.volume.force_index import force_index

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV11(TimeSeriesStrategy):
    """
    策略简介：Supertrend方向 + ADX趋势强度过滤 + Force Index量价确认的4h趋势策略。

    使用指标：
    - Supertrend(10, 3.0): 趋势方向判断
    - ADX(14): 趋势强度过滤，>25=有效趋势
    - Force Index(13): 量价合力方向

    进场条件（做多）：Supertrend=1 且 ADX>threshold 且 Force Index>0
    进场条件（做空）：Supertrend=-1 且 ADX>threshold 且 Force Index<0

    出场条件：
    - ATR追踪止损
    - 分层止盈
    - Supertrend反转 或 ADX跌破threshold

    优点：三重过滤确保趋势强度和方向一致
    缺点：ADX滞后可能延迟入场
    """
    name = "i_alltime_v11"
    warmup = 400
    freq = "4h"

    st_period: int = 10
    st_mult: float = 3.0
    adx_period: int = 14
    adx_threshold: float = 25.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._st_dir = None
        self._adx = None
        self._fi = None
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
        self._adx, _, _ = adx(highs, lows, closes, period=self.adx_period)
        self._fi = force_index(closes, volumes, 13)

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
        adx_val = self._adx[i]
        fi_val = self._fi[i]
        if np.isnan(st_dir) or np.isnan(adx_val) or np.isnan(fi_val):
            return

        self.bars_since_last_scale += 1

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

        if side != 0 and self.entry_price > 0:
            profit_atr = ((price - self.entry_price) / atr_val) if side == 1 else ((self.entry_price - price) / atr_val)
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                cl = max(1, lots // 3)
                if side == 1: context.close_long(lots=cl)
                else: context.close_short(lots=cl)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                cl = max(1, lots // 3)
                if side == 1: context.close_long(lots=cl)
                else: context.close_short(lots=cl)
                self._took_profit_3atr = True
                return

        if side == 1 and (st_dir != 1 or adx_val < self.adx_threshold * 0.8):
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and (st_dir != -1 or adx_val < self.adx_threshold * 0.8):
            context.close_short()
            self._reset_state()
            return

        if side == 0:
            if st_dir == 1 and adx_val > self.adx_threshold and fi_val > 0:
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
            elif st_dir == -1 and adx_val > self.adx_threshold and fi_val < 0:
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
        elif side != 0 and self._should_add(price, atr_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if side == 1: context.buy(add_lots)
                else: context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val):
        if self.position_scale >= MAX_SCALE: return False
        if self.bars_since_last_scale < 10: return False
        if self.direction == 1 and price < self.entry_price + atr_val: return False
        if self.direction == -1 and price > self.entry_price - atr_val: return False
        return True

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_stop_mult * atr_val * spec.multiplier
        if stop_dist <= 0: return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = context.close_raw * spec.multiplier * spec.margin_rate
        if margin <= 0: return 0
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
