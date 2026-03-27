import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.kmeans_regime import kmeans_regime
from indicators.regime.fractal_dimension import fractal_dim
from indicators.trend.supertrend import supertrend
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV116(TimeSeriesStrategy):
    """
    策略简介：K-Means聚类识别趋势集群 + 分形维度确认 + Supertrend方向跟踪。

    使用指标：
    - K-Means Regime(120, 3): 基于RSI/ADX/ATR特征矩阵聚类行情状态
    - Fractal Dimension(60): 分形维度，低值=光滑趋势，高值=噪音
    - Supertrend(10, 3.0): 趋势方向信号
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - K-Means聚类在趋势集群（ADX高、方向性强）
    - 分形维度<1.4（光滑趋势，非随机游走）
    - Supertrend方向=1（上升趋势）
    进场条件（做空）：
    - K-Means聚类在趋势集群
    - 分形维度<1.4
    - Supertrend方向=-1（下降趋势）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 分形维度升高或Supertrend反转

    优点：ML聚类自适应行情分类，分形维度量化趋势质量
    缺点：K-Means需要重训练，集群标签可能不稳定
    """
    name = "ag_alltime_v116"
    warmup = 500
    freq = "4h"

    km_period: int = 120          # Optuna: 80-200
    fd_period: int = 60           # Optuna: 30-90
    fd_thresh: float = 1.4        # Optuna: 1.3-1.5
    st_period: int = 10           # Optuna: 7-15
    st_mult: float = 3.0          # Optuna: 2.0-4.0
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._km_labels = None
        self._km_dist = None
        self._fd = None
        self._st_dir = None
        self._atr = None
        self._avg_volume = None
        self._cluster_adx_mean = None  # Mean ADX per cluster for labeling

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

        # Features for K-Means
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx(highs, lows, closes, period=14)
        atr_arr = self._atr.copy()
        # Normalize ATR by close for scale invariance
        atr_norm = np.full(n, np.nan)
        valid = closes > 1e-9
        atr_norm[valid] = atr_arr[valid] / closes[valid]

        features = np.column_stack([rsi_arr, adx_arr, atr_norm])
        self._km_labels, self._km_dist = kmeans_regime(
            features, period=self.km_period, n_clusters=3
        )

        # Identify which cluster is "trending" by highest average ADX
        # Compute per-cluster mean ADX
        self._cluster_adx_mean = np.full(3, 0.0)
        for c in range(3):
            mask = self._km_labels == c
            valid_adx = adx_arr[mask & ~np.isnan(adx_arr)]
            if len(valid_adx) > 0:
                self._cluster_adx_mean[c] = np.mean(valid_adx)

        self._fd = fractal_dim(closes, period=self.fd_period)
        _, self._st_dir = supertrend(
            highs, lows, closes, period=self.st_period, multiplier=self.st_mult
        )

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def _is_trending_cluster(self, label):
        """Check if current cluster is the one with highest average ADX."""
        if np.isnan(label):
            return False
        lbl = int(label)
        if lbl < 0 or lbl >= 3:
            return False
        return lbl == int(np.argmax(self._cluster_adx_mean))

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
        km_lbl = self._km_labels[i]
        fd_val = self._fd[i]
        st_dir = self._st_dir[i]
        if np.isnan(km_lbl) or np.isnan(fd_val) or np.isnan(st_dir):
            return

        trending = self._is_trending_cluster(km_lbl)
        smooth = fd_val < self.fd_thresh

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

        # 3. Signal-based exit: fractal dim too high or supertrend reversal
        if side == 1 and (fd_val > 1.6 or st_dir == -1):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (fd_val > 1.6 or st_dir == 1):
            context.close_short()
            self._reset_state()
            return

        # Re-read position
        side, lots = context.position

        # 4. Entry: trending cluster + smooth fractal + supertrend direction
        if side == 0 and trending and smooth:
            if st_dir == 1:
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
            elif st_dir == -1:
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
            signal_ok = (self.direction == 1 and st_dir == 1 and trending) or \
                        (self.direction == -1 and st_dir == -1 and trending)
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
