import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi
from indicators.momentum.roc import rate_of_change
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _pca_component(features_matrix, period=60):
    """Rolling PCA: extract dominant component direction.

    Returns (pc1_score, explained_ratio) arrays.
    pc1_score > 0 = dominant component is bullish, < 0 = bearish.
    """
    n = features_matrix.shape[0]
    n_feat = features_matrix.shape[1]
    pc1_score = np.full(n, np.nan, dtype=np.float64)
    explained = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return pc1_score, explained

    for i in range(period, n):
        window = features_matrix[i - period + 1:i + 1, :]
        # Check for NaN
        if np.any(np.isnan(window)):
            continue

        # Standardize
        mu = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        std[std < 1e-12] = 1.0
        X = (window - mu) / std

        # Covariance matrix
        cov = X.T @ X / (period - 1)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue

        # Sort by eigenvalue (descending)
        idx_sort = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_sort]
        eigenvectors = eigenvectors[:, idx_sort]

        # PC1 score: project current standardized observation onto PC1
        current_std = (features_matrix[i] - mu) / std
        if np.any(np.isnan(current_std)):
            continue
        pc1_score[i] = np.dot(current_std, eigenvectors[:, 0])

        total_var = np.sum(eigenvalues)
        if total_var > 0:
            explained[i] = eigenvalues[0] / total_var

    return pc1_score, explained


class StrategyV8(TimeSeriesStrategy):
    """
    策略简介：基于PCA主成分方向的多空交易策略。

    使用指标：
    - PCA(60): 从多特征中提取主成分方向
    - RSI(14), ROC(20), ADX(14): 输入特征
    - ATR(14): 止损距离计算

    进场条件（做多）：PC1得分>1.0, 解释率>0.4
    进场条件（做空）：PC1得分<-1.0, 解释率>0.4

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - PC1得分反转

    优点：降维提取市场核心驱动力，减少噪声
    缺点：PCA对输入特征敏感，解释率低时信号不可靠
    """
    name = "ag_alltime_v8"
    warmup = 300
    freq = "daily"

    pca_period: int = 60          # Optuna: 30-120
    score_thresh: float = 1.0    # Optuna: 0.5-2.0
    explain_thresh: float = 0.4  # Optuna: 0.3-0.6
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._pc1_score = None
        self._explained = None
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

        # Build feature matrix for PCA
        rsi_arr = rsi(closes, period=14)
        roc_arr = rate_of_change(closes, period=20)
        adx_arr = adx(highs, lows, closes, period=14)

        # Price momentum: 5-bar return
        ret5 = np.full_like(closes, np.nan)
        ret5[5:] = closes[5:] / closes[:-5] - 1.0

        features = np.column_stack([rsi_arr, roc_arr, adx_arr, ret5])
        self._pc1_score, self._explained = _pca_component(features, period=self.pca_period)

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
        pc1 = self._pc1_score[i]
        expl = self._explained[i]
        if np.isnan(pc1) or np.isnan(expl):
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
        if side == 1 and pc1 < 0:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and pc1 > 0:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if pc1 > self.score_thresh and expl > self.explain_thresh:
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
            elif pc1 < -self.score_thresh and expl > self.explain_thresh:
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
            if self.direction == 1 and pc1 > self.score_thresh:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and pc1 < -self.score_thresh:
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
