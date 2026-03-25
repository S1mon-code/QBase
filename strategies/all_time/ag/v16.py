import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.trend.adx import adx_with_di

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _hmm_regime(closes, period=120):
    """HMM-like regime detection. Returns (states, prob)."""
    n = len(closes)
    states = np.full(n, 0, dtype=np.int32)
    prob = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return states, prob
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-9))
    for i in range(period, n):
        w = log_ret[i - period + 1:i + 1]
        v = w[~np.isnan(w)]
        if len(v) < 20:
            continue
        mu = np.mean(v)
        sigma = np.std(v)
        if sigma < 1e-12:
            continue
        z = mu / sigma * np.sqrt(len(v))
        if z > 1.0:
            states[i] = 1
            prob[i] = min(1.0, abs(z) / 3.0)
        elif z < -1.0:
            states[i] = 2
            prob[i] = min(1.0, abs(z) / 3.0)
        else:
            states[i] = 0
            prob[i] = 1.0 - abs(z)
    return states, prob


def _trend_strength(adx_arr, plus_di, minus_di):
    """Trend strength score combining ADX and DI spread.

    Returns score in [-1, 1]. Positive = strong uptrend, negative = strong downtrend.
    """
    n = len(adx_arr)
    score = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        a = adx_arr[i]
        pdi = plus_di[i]
        mdi = minus_di[i]
        if np.isnan(a) or np.isnan(pdi) or np.isnan(mdi):
            continue
        # Normalize ADX to 0-1 (ADX typically 0-60)
        adx_norm = min(a / 50.0, 1.0)
        # Direction from DI
        di_total = pdi + mdi
        if di_total > 0:
            direction = (pdi - mdi) / di_total
        else:
            direction = 0.0
        score[i] = adx_norm * direction
    return score


class StrategyV16(TimeSeriesStrategy):
    """
    策略简介：HMM状态 + ADX趋势强度双重确认的多空策略。

    使用指标：
    - HMM Regime(120): 识别市场隐状态（上涨/下跌/震荡）
    - ADX(14) + DI: 趋势强度和方向评分
    - ATR(14): 止损距离计算

    进场条件（做多）：HMM状态=1, HMM置信度>0.3, 趋势强度>0.3
    进场条件（做空）：HMM状态=2, HMM置信度>0.3, 趋势强度<-0.3

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - HMM状态或趋势强度反转

    优点：双重过滤减少假信号，HMM捕捉宏观状态，ADX确认微观趋势
    缺点：双重条件导致信号较少，可能错过快速行情
    """
    name = "ag_alltime_v16"
    warmup = 400
    freq = "4h"

    hmm_period: int = 120         # Optuna: 60-200
    confidence_thresh: float = 0.3  # Optuna: 0.2-0.5
    strength_thresh: float = 0.3  # Optuna: 0.2-0.5
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._hmm_states = None
        self._hmm_prob = None
        self._trend_str = None
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
        self._hmm_states, self._hmm_prob = _hmm_regime(closes, period=self.hmm_period)
        adx_arr, pdi, mdi = adx_with_di(highs, lows, closes, period=14)
        self._trend_str = _trend_strength(adx_arr, pdi, mdi)

        window = 20
        self._avg_volume = np.full_like(volumes, np.nan)
        for idx in range(window, len(volumes)):
            self._avg_volume[idx] = np.mean(volumes[idx - window:idx])

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
        state = self._hmm_states[i]
        prob = self._hmm_prob[i]
        ts = self._trend_str[i]
        if np.isnan(prob) or np.isnan(ts):
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

        # 3. Signal-based exit: either indicator reverses
        if side == 1 and (state == 2 or ts < -0.1):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (state == 1 or ts > 0.1):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry: both must confirm
        if side == 0:
            if state == 1 and prob > self.confidence_thresh and ts > self.strength_thresh:
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
            elif state == 2 and prob > self.confidence_thresh and ts < -self.strength_thresh:
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
            if self.direction == 1 and state == 1 and ts > self.strength_thresh:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and state == 2 and ts < -self.strength_thresh:
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
