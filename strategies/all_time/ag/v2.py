import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.trend.ema import ema

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _kmeans_regime(closes, volumes, period=120, n_clusters=3):
    """Simple K-Means-like regime via rolling return/vol quantiles.

    Returns (cluster_id, centroid_drift) arrays.
    cluster 0 = low vol sideways, 1 = bullish, 2 = bearish.
    centroid_drift > 0 = bullish drift, < 0 = bearish drift.
    """
    n = len(closes)
    cluster = np.full(n, 0, dtype=np.int32)
    drift = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return cluster, drift

    log_ret = np.full(n, 0.0)
    log_ret[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-9))

    for i in range(period, n):
        window_ret = log_ret[i - period + 1:i + 1]
        mu = np.mean(window_ret)
        sigma = np.std(window_ret)
        if sigma < 1e-12:
            cluster[i] = 0
            drift[i] = 0.0
            continue

        cum_ret = np.sum(window_ret)
        # Classify
        if cum_ret > sigma * np.sqrt(period) * 0.5:
            cluster[i] = 1
        elif cum_ret < -sigma * np.sqrt(period) * 0.5:
            cluster[i] = 2
        else:
            cluster[i] = 0

        # Centroid drift = recent half vs first half return
        half = period // 2
        recent = np.mean(window_ret[-half:])
        early = np.mean(window_ret[:half])
        drift[i] = (recent - early) / max(sigma, 1e-12)

    return cluster, drift


class StrategyV2(TimeSeriesStrategy):
    """
    策略简介：基于K-Means聚类质心漂移方向的趋势跟踪策略。

    使用指标：
    - K-Means Regime(120): 聚类识别市场状态，质心漂移方向判断趋势
    - EMA(50): 趋势方向过滤
    - ATR(14): 止损距离计算

    进场条件（做多）：聚类=1(看涨)，质心漂移>0.5，价格>EMA50
    进场条件（做空）：聚类=2(看跌)，质心漂移<-0.5，价格<EMA50

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 聚类状态转为反向

    优点：利用聚类自适应识别市场状态，质心漂移提供方向信号
    缺点：聚类结果可能不稳定，在频繁切换时产生抖动
    """
    name = "ag_alltime_v2"
    warmup = 400
    freq = "daily"

    kmeans_period: int = 120      # Optuna: 60-200
    ema_period: int = 50          # Optuna: 20-80
    drift_thresh: float = 0.5    # Optuna: 0.2-1.0
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._cluster = None
        self._drift = None
        self._ema = None
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
        self._ema = ema(closes, period=self.ema_period)
        self._cluster, self._drift = _kmeans_regime(closes, volumes, period=self.kmeans_period)

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

        ema_val = self._ema[i]
        cluster_val = self._cluster[i]
        drift_val = self._drift[i]
        if np.isnan(drift_val):
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
        if side == 1 and (cluster_val == 2 or drift_val < -self.drift_thresh):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (cluster_val == 1 or drift_val > self.drift_thresh):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if cluster_val == 1 and drift_val > self.drift_thresh and price > ema_val:
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
            elif cluster_val == 2 and drift_val < -self.drift_thresh and price < ema_val:
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
            if self.direction == 1 and cluster_val == 1 and drift_val > 0:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and cluster_val == 2 and drift_val < 0:
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
