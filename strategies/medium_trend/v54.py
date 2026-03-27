import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.regime.trend_persistence import trend_persistence
from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV54(TimeSeriesStrategy):
    """
    策略简介：Trend Persistence持续性检测 + Schaff Trend Cycle动量的30min策略。

    使用指标：
    - Trend Persistence(20, 60): 趋势持续性>0.5时市场处于趋势状态
    - Schaff Trend Cycle(10, 23, 50): >75为多头超买区（但在趋势中持续高位是正常的）
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Trend Persistence > tp_threshold（确认趋势持续中）
    - STC > 25且从低位上穿25（趋势启动）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - STC < 25（趋势结束信号）

    优点：趋势持续性是强有力的过滤器，STC比MACD更灵敏
    缺点：Trend Persistence计算依赖足够数据，STC可能过于灵敏
    """
    name = "medium_trend_v54"
    warmup = 700
    freq = "30min"

    tp_max_lag: int = 20
    tp_period: int = 60
    tp_threshold: float = 0.5         # Optuna: 0.3-0.7
    stc_period: int = 10
    stc_fast: int = 23
    stc_slow: int = 50
    atr_stop_mult: float = 3.0       # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._tp = None
        self._stc = None
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

        self._tp = trend_persistence(closes, max_lag=self.tp_max_lag, period=self.tp_period)
        self._stc = schaff_trend_cycle(closes, period=self.stc_period, fast=self.stc_fast, slow=self.stc_slow)
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
        if i < 1:
            return

        atr_val = self._atr[i]
        tp_val = self._tp[i]
        stc_now = self._stc[i]
        stc_prev = self._stc[i - 1]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(tp_val) or np.isnan(stc_now) or np.isnan(stc_prev):
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

        if side == 1 and stc_now < 25:
            context.close_long()
            self._reset_state()
            return

        # Entry: STC crosses above 25 + trend is persistent
        if side == 0 and tp_val > self.tp_threshold and stc_now > 25 and stc_prev <= 25:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, tp_val, stc_now):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, tp_val, stc_now):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if tp_val < self.tp_threshold or stc_now < 25:
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
