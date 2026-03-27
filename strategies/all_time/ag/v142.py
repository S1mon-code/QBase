import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.kmeans_regime import kmeans_regime
from indicators.regime.trend_persistence import trend_persistence
from indicators.microstructure.volume_imbalance import volume_imbalance
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class SVMTrendPersistenceVolumeImbalanceStrategy(TimeSeriesStrategy):
    """
    策略简介：K-Means 聚类（代替 SVM）检测趋势集群 + 趋势持续性确认 + 成交量买卖压力失衡的 1h 多空策略。

    使用指标：
    - K-Means Regime(3 clusters, 120): 将市场状态聚类为趋势/震荡/过渡态
    - Trend Persistence(20, 60): 自相关性度量趋势持续性，高值=强趋势
    - Volume Imbalance(20): 买卖压力失衡，正值=买入压力，负值=卖出压力
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - K-Means 聚类为趋势状态
    - 趋势持续性 > threshold（强持续趋势）
    - Volume imbalance > 0（净买入压力）

    进场条件（做空）：
    - K-Means 聚类为趋势状态
    - 趋势持续性 > threshold
    - Volume imbalance < 0（净卖出压力）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 趋势持续性下降或 volume imbalance 反转

    优点：多维度确认趋势有效性 + 成交量微观结构提供额外验证
    缺点：K-Means 不稳定可能导致 regime 频繁跳变，1h 频率噪音较大
    """
    name = "v142_svm_trend_persistence_volume_imbalance"
    warmup = 500
    freq = "1h"

    km_period: int = 120
    tp_period: int = 60
    vi_period: int = 20
    persistence_threshold: float = 0.3
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._km_labels = None
        self._persistence = None
        self._vi_imbalance = None
        self._vi_signal = None
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

        # Build features for K-Means
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx(highs, lows, closes, period=14)
        atr_arr = atr(highs, lows, closes, period=14)
        features = np.column_stack([rsi_arr, adx_arr, atr_arr])
        self._km_labels, _ = kmeans_regime(features, period=self.km_period, n_clusters=3)

        self._persistence, _ = trend_persistence(closes, max_lag=20, period=self.tp_period)
        self._vi_imbalance, self._vi_signal = volume_imbalance(closes, volumes, period=self.vi_period)
        self._atr = atr_arr

        # Identify which cluster is "trending" by highest average ADX
        # We'll compute average ADX per cluster label for non-NaN values
        self._trend_cluster = 1  # default
        valid = ~np.isnan(self._km_labels) & ~np.isnan(adx_arr)
        if np.sum(valid) > 100:
            labels_valid = self._km_labels[valid].astype(int)
            adx_valid = adx_arr[valid]
            avg_adx = [np.mean(adx_valid[labels_valid == c]) if np.sum(labels_valid == c) > 0 else 0
                       for c in range(3)]
            self._trend_cluster = int(np.argmax(avg_adx))

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        km_label = self._km_labels[i]
        persist = self._persistence[i]
        vi_val = self._vi_signal[i]
        atr_val = self._atr[i]
        if np.isnan(km_label) or np.isnan(atr_val):
            return
        if np.isnan(persist):
            persist = 0.0
        if np.isnan(vi_val):
            vi_val = 0.0

        self.bars_since_last_scale += 1
        is_trend_cluster = int(km_label) == self._trend_cluster
        is_persistent = persist > self.persistence_threshold

        # ── 1. 止损检查 ──
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

        # ── 2. 分层止盈 ──
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
        elif side == -1 and self.entry_price > 0:
            profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_3atr = True
                return

        # ── 3. 信号弱化退出 ──
        if side == 1 and (not is_persistent or vi_val < -0.2):
            context.close_long()
            self._reset_state()
        elif side == -1 and (not is_persistent or vi_val > 0.2):
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if is_trend_cluster and is_persistent and vi_val > 0:
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
            elif is_trend_cluster and is_persistent and vi_val < 0:
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

        # ── 5. 加仓逻辑 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and is_trend_cluster and vi_val > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and is_trend_cluster and vi_val < 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.sell(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
