import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.nearest_neighbor_signal import knn_signal
from indicators.regime.vol_regime_markov import vol_regime_simple
from indicators.volatility.hurst import hurst_exponent

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV140(TimeSeriesStrategy):
    """
    策略简介：KNN近邻回归方向预测 + 波动率状态扩张 + Hurst趋势确认的多空策略。

    使用指标：
    - KNN Signal(120, 5): K近邻回归预测未来收益方向，>0做多，<0做空
    - Vol Regime Simple(60): 波动率状态(0=低波/1=高波)，高波态=扩张有利趋势
    - Hurst Exponent(20): >0.55=趋势性，<0.45=均值回归，过滤趋势环境

    进场条件（做多）：KNN预测>0 + 波动率扩张态(regime=1) + Hurst>0.55
    进场条件（做空）：KNN预测<0 + 波动率扩张态(regime=1) + Hurst>0.55

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - KNN预测反向或Hurst<0.45(失去趋势性)

    优点：KNN非参数化适应任何分布，Hurst直接度量趋势持续性
    缺点：KNN对特征缩放敏感，Hurst估计短窗口下噪声大
    """
    name = "ag_alltime_v140"
    warmup = 500
    freq = "daily"

    knn_period: int = 120          # Optuna: 60-200
    knn_k: int = 5                 # Optuna: 3-10
    vol_period: int = 60           # Optuna: 30-100
    hurst_lag: int = 20            # Optuna: 10-40
    hurst_thresh: float = 0.55     # Optuna: 0.50-0.65
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._knn_pred = None
        self._knn_conf = None
        self._vol_regime = None
        self._hurst = None
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

        # KNN signal
        from indicators.momentum.rsi import rsi
        from indicators.trend.adx import adx
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx(highs, lows, closes, period=14)
        features = np.column_stack([rsi_arr, adx_arr, self._atr, volumes])
        self._knn_pred, self._knn_conf = knn_signal(
            closes, features, period=self.knn_period, k=self.knn_k)

        # Vol regime
        self._vol_regime, _, _ = vol_regime_simple(closes, period=self.vol_period)

        # Hurst exponent
        self._hurst = hurst_exponent(closes, max_lag=self.hurst_lag)

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
        knn_val = self._knn_pred[i]
        vr_val = self._vol_regime[i]
        hurst_val = self._hurst[i]
        if np.isnan(knn_val) or np.isnan(vr_val) or np.isnan(hurst_val):
            return

        self.bars_since_last_scale += 1

        vol_expanding = vr_val == 1
        is_trending = hurst_val > self.hurst_thresh

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

        # 3. Signal-based exit: KNN flip or Hurst mean reverting
        if side == 1 and (knn_val < 0 or hurst_val < 0.45):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (knn_val > 0 or hurst_val < 0.45):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and vol_expanding and is_trending:
            if knn_val > 0:
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
            elif knn_val < 0:
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
            if self.direction == 1 and knn_val > 0 and is_trending:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and knn_val < 0 and is_trending:
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
