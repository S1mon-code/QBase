import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.ml.kmeans_regime import kmeans_regime
from indicators.momentum.macd import macd
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV83(TimeSeriesStrategy):
    """
    策略简介：K-Means聚类识别行情状态 + MACD动量确认的4h多头策略。

    使用指标：
    - K-Means Regime(120, 3): 基于多特征聚类识别趋势/震荡状态
    - MACD(12,26,9): 动量方向确认和信号线交叉
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - K-Means聚类标签处于趋势上升状态（cluster距心最高收益率对应的cluster）
    - MACD柱状图 > 0且MACD线 > 信号线

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - MACD柱状图转负

    优点：ML状态识别过滤震荡，MACD确认动量
    缺点：K-Means可能不稳定，cluster标签在不同窗口可能漂移
    """
    name = "medium_trend_v83"
    warmup = 400
    freq = "4h"

    kmeans_period: int = 120     # Optuna: 80-200
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_stop_mult: float = 3.0  # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._regime_labels = None
        self._regime_centers = None
        self._macd_line = None
        self._macd_signal = None
        self._macd_hist = None
        self._atr = None
        self._avg_volume = None
        self._bullish_cluster = None

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

        self._atr = atr(highs, lows, closes, period=14)
        self._macd_line, self._macd_signal, self._macd_hist = macd(
            closes, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal
        )

        # Build features for K-Means: ROC, ATR ratio, volume ratio
        n = len(closes)
        roc_20 = np.full(n, np.nan)
        roc_20[20:] = (closes[20:] - closes[:-20]) / closes[:-20]
        atr_arr = self._atr.copy()
        atr_ratio = np.full(n, np.nan)
        valid = closes > 0
        atr_ratio[valid] = atr_arr[valid] / closes[valid]
        vol_ma = fast_avg_volume(volumes, 20)
        vol_ratio = np.full(n, np.nan)
        valid_v = vol_ma > 0
        vol_ratio[valid_v] = volumes[valid_v] / vol_ma[valid_v]

        features = np.column_stack([roc_20, atr_ratio, vol_ratio])
        self._regime_labels, self._regime_centers = kmeans_regime(
            features, period=self.kmeans_period, n_clusters=3
        )

        # Identify the bullish cluster: the one with highest avg ROC
        # We do this by looking at average returns per cluster over the whole series
        cluster_returns = {}
        for c in range(3):
            mask = self._regime_labels == c
            if np.sum(mask) > 10:
                cluster_returns[c] = np.nanmean(roc_20[mask])
            else:
                cluster_returns[c] = -999.0
        self._bullish_cluster = max(cluster_returns, key=cluster_returns.get)

        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        regime = self._regime_labels[i]
        hist = self._macd_hist[i]
        ml = self._macd_line[i]
        ms = self._macd_signal[i]
        if np.isnan(regime) or np.isnan(hist) or np.isnan(ml) or np.isnan(ms):
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

        # 2. Tiered profit-taking
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

        # 3. Signal exit: MACD histogram turns negative
        if side == 1 and hist < 0:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry: bullish cluster + MACD crossover
        in_bullish = int(regime) == self._bullish_cluster
        if side == 0 and in_bullish and hist > 0 and ml > ms:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, in_bullish, hist):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, in_bullish, hist):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if not in_bullish or hist <= 0:
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
