import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.ml.hmm_regime import hmm_regime
from indicators.regime.trend_strength_composite import trend_strength
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV103(TimeSeriesStrategy):
    """
    策略简介：HMM Regime隐马尔可夫状态识别 + Trend Strength确认的日线多头策略。

    使用指标：
    - HMM Regime(3, 252): 隐马尔可夫模型识别3个行情状态
    - Trend Strength(20): 综合趋势强度评分确认
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - HMM处于上升状态（对应最高收益率的state）
    - Trend Strength > ts_threshold

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - HMM切换到非上升状态

    优点：HMM考虑状态转移概率，比简单聚类更合理
    缺点：HMM训练不稳定，状态标签可能翻转
    """
    name = "medium_trend_v103"
    warmup = 120
    freq = "daily"

    hmm_states: int = 3
    hmm_period: int = 252       # Optuna: 180-300
    ts_period: int = 20         # Optuna: 14-30
    ts_threshold: float = 0.4   # Optuna: 0.2-0.6
    atr_stop_mult: float = 3.0  # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._hmm_labels = None
        self._ts = None
        self._atr = None
        self._avg_volume = None
        self._bullish_state = 0

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

        self._hmm_labels, _, _ = hmm_regime(
            closes, n_states=self.hmm_states, period=self.hmm_period
        )
        self._ts = trend_strength(closes, highs, lows, period=self.ts_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        # Identify bullish state
        n = len(closes)
        roc = np.full(n, np.nan)
        roc[10:] = (closes[10:] - closes[:-10]) / closes[:-10]
        state_avg = {}
        for s in range(self.hmm_states):
            mask = self._hmm_labels == s
            if np.sum(mask) > 5:
                state_avg[s] = np.nanmean(roc[mask])
            else:
                state_avg[s] = -999.0
        self._bullish_state = max(state_avg, key=state_avg.get)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        label = self._hmm_labels[i]
        ts_val = self._ts[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(label) or np.isnan(ts_val):
            return

        in_bullish = int(label) == self._bullish_state
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

        if side == 1 and not in_bullish:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and in_bullish and ts_val > self.ts_threshold:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, in_bullish, ts_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, in_bullish, ts_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if not in_bullish or ts_val < self.ts_threshold:
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
