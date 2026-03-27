import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # QBase root
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _hmm_regime(closes, period=120, n_states=3):
    """Simple HMM-like regime detection via rolling statistics.

    Classifies each bar into one of n_states regimes based on rolling
    return and volatility quantiles. Returns an integer array (0..n_states-1)
    and a state_prob array (confidence 0-1).
    State 0 = low vol / sideways, 1 = trending up, 2 = trending down.
    """
    n = len(closes)
    states = np.full(n, 0, dtype=np.int32)
    state_prob = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return states, state_prob

    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-9))

    for i in range(period, n):
        window = log_ret[i - period + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid)
        if sigma < 1e-12:
            states[i] = 0
            state_prob[i] = 0.5
            continue

        # Z-score of recent return
        z = mu / sigma * np.sqrt(len(valid))

        if z > 1.0:
            states[i] = 1  # trending up
            state_prob[i] = min(1.0, abs(z) / 3.0)
        elif z < -1.0:
            states[i] = 2  # trending down
            state_prob[i] = min(1.0, abs(z) / 3.0)
        else:
            states[i] = 0  # sideways
            state_prob[i] = 1.0 - abs(z)

    return states, state_prob


class StrategyV1(TimeSeriesStrategy):
    """
    策略简介：基于HMM隐状态转换的多空趋势跟踪策略。

    使用指标：
    - HMM Regime(120): 识别市场隐状态（上涨/下跌/震荡）
    - RSI(14): 超买超卖过滤
    - ATR(14): 止损距离计算

    进场条件（做多）：HMM状态=1(上涨)，置信度>0.3，RSI<75
    进场条件（做空）：HMM状态=2(下跌)，置信度>0.3，RSI>25

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - HMM状态转换为反向

    优点：状态识别自适应市场环境，减少震荡市假信号
    缺点：状态切换滞后，短周期趋势可能错过
    """
    name = "ag_alltime_v1"
    warmup = 400  # 120 * 3 + buffer
    freq = "4h"

    hmm_period: int = 120         # Optuna: 60-200
    rsi_period: int = 14          # Optuna: 10-20
    confidence_thresh: float = 0.3  # Optuna: 0.2-0.6
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._hmm_states = None
        self._hmm_prob = None
        self._rsi = None
        self._atr = None
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
        self._rsi = rsi(closes, period=self.rsi_period)
        self._hmm_states, self._hmm_prob = _hmm_regime(closes, period=self.hmm_period)

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        rsi_val = self._rsi[i]
        if np.isnan(rsi_val):
            return
        state = self._hmm_states[i]
        prob = self._hmm_prob[i]
        if np.isnan(prob):
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
            if side == 1:
                profit_atr = (price - self.entry_price) / atr_val
            else:
                profit_atr = (self.entry_price - price) / atr_val

            if profit_atr >= 5.0 and not self._took_profit_5atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_3atr = True
                return

        # 3. Signal-based exit
        if side == 1 and state == 2:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and state == 1:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry logic
        if side == 0:
            if state == 1 and prob > self.confidence_thresh and rsi_val < 75:
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
            elif state == 2 and prob > self.confidence_thresh and rsi_val > 25:
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
            if self.direction == 1 and state == 1:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and state == 2:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
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
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.direction = 0
