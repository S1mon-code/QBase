import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.volatility.atr import atr
from indicators.volatility.natr import natr
from indicators.trend.supertrend import supertrend

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class MediumTrendV226(TimeSeriesStrategy):
    """
    策略简介：NATR标准化波动率低位反弹 + Supertrend趋势的1h做多策略。

    使用指标：
    - NATR(14): 标准化ATR，可跨品种比较
    - Supertrend(10, 3.0): 趋势方向
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - NATR处于近40期低位（波动率压缩状态）
    - Supertrend翻多（方向从-1变为+1）

    出场条件：
    - ATR追踪止损（3.0倍ATR）
    - 分层止盈
    - Supertrend翻空

    优点：低NATR入场代表安全区域，趋势启动时获利大
    缺点：低波动不一定会扩张
    """
    name = "mt_v226"
    warmup = 60
    freq = "1h"

    natr_lookback: int = 40
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._natr = None
        self._st_dir = None
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
        self._atr = atr(highs, lows, closes, period=14)
        self._natr = natr(highs, lows, closes)
        _, self._st_dir = supertrend(highs, lows, closes)
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
        if np.isnan(atr_val) or atr_val <= 0:
            return
        natr_val = self._natr[i]
        st_dir = self._st_dir[i]
        if np.isnan(natr_val) or np.isnan(st_dir):
            return
        prev_dir = self._st_dir[i - 1] if i > 0 else np.nan
        if np.isnan(prev_dir):
            return
        self.bars_since_last_scale += 1

        # Check if NATR is in bottom 25% of recent window
        natr_low = False
        if i >= self.natr_lookback:
            natr_win = self._natr[i - self.natr_lookback:i]
            valid = natr_win[~np.isnan(natr_win)]
            if len(valid) > 5:
                natr_low = natr_val <= np.percentile(valid, 25)

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

        if side == 1 and st_dir == -1:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and natr_low and prev_dir == -1 and st_dir == 1:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0
        elif side == 1 and self.position_scale < MAX_SCALE and self.bars_since_last_scale >= 10 and price > self.entry_price + atr_val:
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
