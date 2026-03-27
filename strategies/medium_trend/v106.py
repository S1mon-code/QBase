import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.adx import adx
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV106(TimeSeriesStrategy):
    """
    策略简介：ADX趋势强度过滤 + Supertrend方向信号的日线多头策略。

    使用指标：
    - ADX(14): > threshold确认趋势存在
    - Supertrend(10, 3.0): 方向=1时做多
    - ATR(14): 止损距离

    进场条件（做多）：ADX > threshold且Supertrend = 1
    出场条件：Supertrend翻转为-1 / ATR止损 / 分层止盈

    优点：ADX+Supertrend互补，过滤+方向分工明确
    缺点：两个指标都基于ATR，本质上有一定相关性
    """
    name = "medium_trend_v106"
    warmup = 80
    freq = "daily"

    adx_period: int = 14
    adx_threshold: float = 25.0
    st_period: int = 10
    st_mult: float = 3.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._adx = None
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
        self._adx = adx(highs, lows, closes, period=self.adx_period)
        _, self._st_dir = supertrend(highs, lows, closes, period=self.st_period, multiplier=self.st_mult)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return
        atr_val = self._atr[i]
        adx_val = self._adx[i]
        st_dir = self._st_dir[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(adx_val) or np.isnan(st_dir):
            return
        self.bars_since_last_scale += 1

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long(); self._reset_state(); return
        if side == 1 and self.entry_price > 0:
            profit_atr = (price - self.entry_price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_long(lots=max(1, lots // 3)); self._took_profit_5atr = True; return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_long(lots=max(1, lots // 3)); self._took_profit_3atr = True; return
        if side == 1 and st_dir == -1:
            context.close_long(); self._reset_state(); return
        if side == 0 and adx_val > self.adx_threshold and st_dir == 1:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1; self.bars_since_last_scale = 0
        elif side == 1 and self.position_scale < MAX_SCALE and self.bars_since_last_scale >= 10 and price > self.entry_price + atr_val and st_dir == 1:
            factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
            add = max(1, int(self._calc_lots(context, atr_val) * factor))
            if add > 0:
                context.buy(add); self.position_scale += 1; self.bars_since_last_scale = 0

    def _calc_lots(self, context, atr_val):
        from alphaforge.data.contract_specs import ContractSpecManager
        spec = ContractSpecManager().get(context.symbol)
        stop_dist = self.atr_stop_mult * atr_val * spec.multiplier
        if stop_dist <= 0: return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = context.close_raw * spec.multiplier * spec.margin_rate
        if margin <= 0: return 0
        return max(1, min(risk_lots, int(context.equity * 0.30 / margin)))

    def _reset_state(self):
        self.entry_price = 0.0; self.stop_price = 0.0; self.highest_since_entry = 0.0
        self.position_scale = 0; self.bars_since_last_scale = 0
        self._took_profit_3atr = False; self._took_profit_5atr = False
