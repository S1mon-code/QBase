import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.trend.mesa_adaptive_ma import mama
from indicators.volume.obv import obv
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV72(TimeSeriesStrategy):
    """
    策略简介：MESA Adaptive MA (MAMA/FAMA)交叉 + OBV趋势确认的1h自适应策略。

    使用指标：
    - MAMA(0.5, 0.05): MAMA > FAMA为多头
    - OBV: OBV斜率>0表示资金持续流入
    - ATR(14): 止损距离

    进场条件（做多）：MAMA > FAMA 且 OBV上升
    出场条件：追踪止损 / 分层止盈 / MAMA < FAMA

    优点：MAMA自适应频率，比固定周期均线更灵活
    缺点：MESA算法参数敏感，OBV计算简单可能噪音大
    """
    name = "medium_trend_v72"
    warmup = 400
    freq = "1h"

    mama_fast: float = 0.5
    mama_slow: float = 0.05
    obv_slope_period: int = 20        # Optuna: 10-30
    atr_stop_mult: float = 3.0       # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._mama_line = None
        self._fama_line = None
        self._obv = None
        self._obv_slope = None
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

        result = mama(closes, fast_limit=self.mama_fast, slow_limit=self.mama_slow)
        self._mama_line = result[0]
        self._fama_line = result[1]
        self._obv = obv(closes, volumes)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        n = len(closes)
        p = self.obv_slope_period
        self._obv_slope = np.full(n, np.nan)
        for j in range(p, n):
            seg = self._obv[j - p:j]
            if not np.any(np.isnan(seg)):
                x = np.arange(p, dtype=np.float64)
                self._obv_slope[j] = np.polyfit(x, seg, 1)[0]

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        m = self._mama_line[i]
        f = self._fama_line[i]
        os = self._obv_slope[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(m) or np.isnan(f) or np.isnan(os):
            return

        bull = m > f
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

        if side == 1 and not bull:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and bull and os > 0:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, bull, os):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, bull, os):
        if self.position_scale >= MAX_SCALE or self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        return bull and os > 0

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
