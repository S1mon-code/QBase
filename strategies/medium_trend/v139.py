import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.mesa_adaptive_ma import mama
from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV139(TimeSeriesStrategy):
    """
    策略简介：4h MESA Adaptive MA方向 + 5min Schaff Trend Cycle入场。

    使用指标：
    - MAMA [4h]: 自适应均线，price > mama = 上升趋势
    - Schaff Trend Cycle(10,23,50) [5min]: STC从<25上穿25入场
    - ATR(14) [5min]: 止损距离

    进场条件（做多）：4h close > MAMA, 5min STC上穿25
    出场条件：ATR追踪止损, 分层止盈, close < MAMA

    优点：MAMA自适应周期，STC结合趋势和周期
    缺点：MAMA参数不透明
    """
    name = "medium_trend_v139"
    freq = "5min"
    warmup = 2000

    stc_entry: float = 25.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._stc = None
        self._atr = None
        self._avg_volume = None
        self._mama_4h = None
        self._closes_4h = None
        self._4h_map = None

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
        n = len(closes)

        self._stc = schaff_trend_cycle(closes, period=10, fast=23, slow=50)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 48
        n_4h = n // step
        trim = n_4h * step
        closes_4h = closes[:trim].reshape(n_4h, step)[:, -1]

        self._mama_4h = mama(closes_4h)
        self._closes_4h = closes_4h
        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1),
                                  len(self._mama_4h) - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        stc_val = self._stc[i]
        atr_val = self._atr[i]
        mama_val = self._mama_4h[j]
        close_4h = self._closes_4h[j]
        if np.isnan(stc_val) or np.isnan(atr_val) or np.isnan(mama_val):
            return

        prev_stc = self._stc[i - 1] if i > 0 else np.nan
        above_mama = close_4h > mama_val
        stc_cross_up = (not np.isnan(prev_stc) and prev_stc <= self.stc_entry and stc_val > self.stc_entry)
        self.bars_since_last_scale += 1

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

        if side == 1 and not above_mama:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and above_mama and stc_cross_up:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and above_mama and stc_val > 50):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _calc_lots(self, context, atr_val):
        from alphaforge.data.contract_specs import ContractSpecManager
        spec = ContractSpecManager().get(context.symbol)
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
