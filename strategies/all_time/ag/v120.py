import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.regime_transition_matrix import transition_features
from indicators.ml.hmm_regime import hmm_regime
from indicators.volume.volume_spike import volume_spike
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _simple_nn_signal(closes, features_matrix, period=120):
    """Simple neural network direction signal using MLPClassifier.

    Trains a small MLP on trailing window to predict direction.
    Retrains every period//4 bars.

    Returns probability of positive return (0-1).
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    n = len(closes)
    signal = np.full(n, np.nan, dtype=np.float64)

    if n < period + 2:
        return signal

    safe = np.maximum(closes, 1e-12)
    fwd_dir = np.full(n, np.nan, dtype=np.float64)
    fwd_dir[:-1] = np.where(safe[1:] > safe[:-1], 1.0, 0.0)

    retrain_every = max(1, period // 4)
    model = None
    scaler = None

    for i in range(period, n):
        need_train = model is None or (i - period) % retrain_every == 0

        if need_train:
            X_train = features_matrix[i - period:i]
            y_train = fwd_dir[i - period:i]
            valid_mask = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
            X_valid = X_train[valid_mask]
            y_valid = y_train[valid_mask]

            if len(X_valid) < 30 or len(np.unique(y_valid)) < 2:
                continue

            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_valid)
                model = MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    max_iter=100, random_state=42,
                    early_stopping=True, validation_fraction=0.2,
                    n_iter_no_change=10,
                )
                model.fit(X_scaled, y_valid)
            except Exception:
                model = None
                continue

        if model is not None and scaler is not None:
            x_cur = features_matrix[i:i + 1]
            if not np.any(np.isnan(x_cur)):
                try:
                    x_scaled = scaler.transform(x_cur)
                    proba = model.predict_proba(x_scaled)
                    # probability of class 1.0 (up)
                    if proba.shape[1] == 2:
                        signal[i] = proba[0, 1]
                    else:
                        signal[i] = proba[0, 0]
                except Exception:
                    pass

    return signal


class StrategyV120(TimeSeriesStrategy):
    """
    策略简介：神经网络方向预测 + HMM Regime转换 + 成交量放大确认。

    使用指标：
    - Neural Network (MLP, 120): 基于RSI/ADX/ATR/ROC特征预测方向
    - HMM Regime(252): 隐马尔可夫模型状态 + Transition Features
    - Volume Spike(20, 2.0): 成交量异常放大检测
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - NN预测上涨概率>0.6
    - HMM regime正在转换（self-transition概率低，regime entropy高）
    - 出现Volume Spike（成交量放大确认方向变化）
    进场条件（做空）：
    - NN预测上涨概率<0.4
    - HMM regime转换中
    - Volume Spike出现

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - NN信号反转或regime稳定不再转换

    优点：NN捕捉复杂非线性关系，regime转换+量能放大精准定时
    缺点：NN训练耗时，regime转换判断可能不稳定
    """
    name = "ag_alltime_v120"
    warmup = 800
    freq = "1h"

    nn_period: int = 120          # Optuna: 80-200
    hmm_period: int = 252         # Optuna: 120-504
    vol_spike_period: int = 20    # Optuna: 10-30
    vol_spike_mult: float = 2.0   # Optuna: 1.5-3.0
    nn_long_thresh: float = 0.6   # Optuna: 0.55-0.70
    nn_short_thresh: float = 0.4  # Optuna: 0.30-0.45
    self_trans_thresh: float = 0.7  # Optuna: 0.5-0.9
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._nn_sig = None
        self._self_trans = None
        self._regime_entropy = None
        self._vol_spike = None
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
        n = len(closes)

        self._atr = atr(highs, lows, closes, period=14)

        # Features for NN
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx(highs, lows, closes, period=14)
        atr_arr = self._atr.copy()
        atr_norm = np.full(n, np.nan)
        valid = closes > 1e-9
        atr_norm[valid] = atr_arr[valid] / closes[valid]

        roc_arr = np.full(n, np.nan)
        for idx in range(10, n):
            if closes[idx - 10] > 1e-9:
                roc_arr[idx] = (closes[idx] / closes[idx - 10]) - 1.0

        features = np.column_stack([rsi_arr, adx_arr, atr_norm, roc_arr])
        self._nn_sig = _simple_nn_signal(closes, features, period=self.nn_period)

        # HMM regime for transition detection
        hmm_labels, _, _ = hmm_regime(closes, n_states=3, period=self.hmm_period)
        self._self_trans, _, self._regime_entropy = transition_features(
            hmm_labels, n_states=3
        )

        # Volume spike
        self._vol_spike = volume_spike(
            volumes, period=self.vol_spike_period, threshold=self.vol_spike_mult
        )

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
        nn_val = self._nn_sig[i]
        self_trans = self._self_trans[i]
        is_spike = self._vol_spike[i] if i < len(self._vol_spike) else False
        if np.isnan(nn_val) or np.isnan(self_trans):
            return

        # Regime transition: self-transition probability low = transitioning
        regime_transitioning = self_trans < self.self_trans_thresh

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

        # 3. Signal-based exit: NN reversal or regime stabilized
        if side == 1 and (nn_val < 0.45 or self_trans > 0.9):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (nn_val > 0.55 or self_trans > 0.9):
            context.close_short()
            self._reset_state()
            return

        # Re-read position
        side, lots = context.position

        # 4. Entry: NN signal + regime transition + volume spike
        if side == 0 and regime_transitioning and is_spike:
            if nn_val > self.nn_long_thresh:
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
            elif nn_val < self.nn_short_thresh:
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
            signal_ok = (self.direction == 1 and nn_val > 0.55) or \
                        (self.direction == -1 and nn_val < 0.45)
            if signal_ok:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    if self.direction == 1:
                        context.buy(add_lots)
                    else:
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
