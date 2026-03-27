import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi
from indicators.momentum.roc import rate_of_change
from indicators.trend.ema import ema
from indicators.trend.linear_regression import linear_regression_slope

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _boosting_signal(features_list, closes, lookback=60):
    """Simple gradient boosting-like ensemble signal from multiple features.

    Each feature votes +1/-1 based on its z-score, then signals are
    weighted by their recent accuracy (boosting idea).
    Returns signal array in [-1, 1].
    """
    n = len(closes)
    n_features = len(features_list)
    signal = np.full(n, np.nan, dtype=np.float64)

    if n < lookback + 1:
        return signal

    # Precompute forward returns for accuracy weighting
    fwd_ret = np.full(n, np.nan)
    fwd_ret[:-1] = closes[1:] / closes[:-1] - 1.0

    for i in range(lookback, n):
        votes = np.zeros(n_features)
        weights = np.ones(n_features)

        for f_idx, feat in enumerate(features_list):
            val = feat[i]
            if np.isnan(val):
                continue

            # Z-score of current value vs lookback window
            window = feat[i - lookback:i]
            valid = window[~np.isnan(window)]
            if len(valid) < 10:
                continue
            z = (val - np.mean(valid)) / max(np.std(valid), 1e-12)

            # Vote
            if z > 0.5:
                votes[f_idx] = 1.0
            elif z < -0.5:
                votes[f_idx] = -1.0

            # Accuracy weighting (boosting): check last 20 predictions
            correct = 0
            total = 0
            for j in range(max(lookback, i - 20), i):
                fj = feat[j]
                if np.isnan(fj) or np.isnan(fwd_ret[j]):
                    continue
                wj = feat[j - lookback:j]
                vj = wj[~np.isnan(wj)]
                if len(vj) < 10:
                    continue
                zj = (fj - np.mean(vj)) / max(np.std(vj), 1e-12)
                pred = 1 if zj > 0 else -1
                actual = 1 if fwd_ret[j] > 0 else -1
                if pred == actual:
                    correct += 1
                total += 1

            if total > 5:
                weights[f_idx] = max(0.1, correct / total)

        total_weight = np.sum(np.abs(weights))
        if total_weight > 0:
            signal[i] = np.sum(votes * weights) / total_weight

    return signal


class StrategyV5(TimeSeriesStrategy):
    """
    策略简介：基于多特征梯度提升集成信号的交易策略。

    使用指标：
    - RSI(14): 动量特征
    - ROC(20): 变化率特征
    - EMA(50): 趋势特征
    - LinReg Slope(20): 回归斜率特征
    - Boosting Ensemble: 加权集成信号
    - ATR(14): 止损距离计算

    进场条件（做多）：集成信号>0.3
    进场条件（做空）：集成信号<-0.3

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 集成信号反转到中性区间

    优点：多特征投票+准确率加权，自适应特征重要性
    缺点：计算复杂，lookback窗口内过拟合风险
    """
    name = "ag_alltime_v5"
    warmup = 300
    freq = "4h"

    lookback: int = 60             # Optuna: 30-120
    signal_thresh: float = 0.3    # Optuna: 0.1-0.5
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._signal = None
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

        # Build feature matrix
        rsi_arr = rsi(closes, period=14)
        roc_arr = rate_of_change(closes, period=20)
        ema_arr = ema(closes, period=50)
        # EMA distance from price as feature
        ema_dist = np.full_like(closes, np.nan)
        valid_ema = ~np.isnan(ema_arr) & (ema_arr > 0)
        ema_dist[valid_ema] = (closes[valid_ema] - ema_arr[valid_ema]) / ema_arr[valid_ema] * 100
        slope_arr = linear_regression_slope(closes, period=20)

        features = [rsi_arr, roc_arr, ema_dist, slope_arr]
        self._signal = _boosting_signal(features, closes, lookback=self.lookback)

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
        sig = self._signal[i]
        if np.isnan(sig):
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
        if side == 1 and sig < 0:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and sig > 0:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if sig > self.signal_thresh:
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
            elif sig < -self.signal_thresh:
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
            if self.direction == 1 and sig > self.signal_thresh:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and sig < -self.signal_thresh:
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
