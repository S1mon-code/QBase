import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.kmeans_regime import kmeans_regime
from indicators.trend.ema import ema_cross

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV136(TimeSeriesStrategy):
    """
    策略简介：K-Means聚类状态 + 状态年龄(新鲜度) + EMA交叉方向的多空策略。

    使用指标：
    - K-Means Regime(120, 3): 多特征聚类分状态，距离中心越近越确定
    - Regime Age: 当前状态持续bar数，新鲜状态(<10bars)信号更可靠
    - EMA Cross(12, 26): 快慢EMA交叉作为入场方向

    进场条件（做多）：EMA金叉 + K-Means状态刚切换(<10bars) + 距离中心近
    进场条件（做空）：EMA死叉 + K-Means状态刚切换(<10bars) + 距离中心近

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - EMA反向交叉或状态老化(>50bars)

    优点：新鲜状态切换捕捉行情转折，EMA交叉提供明确方向
    缺点：K-Means聚类标签不稳定，重训练可能导致标签切换
    """
    name = "ag_alltime_v136"
    warmup = 500
    freq = "4h"

    km_period: int = 120           # Optuna: 60-200
    km_clusters: int = 3           # Optuna: 2-5
    ema_fast: int = 12             # Optuna: 8-20
    ema_slow: int = 26             # Optuna: 20-50
    regime_age_max: int = 10       # Optuna: 5-20
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._km_labels = None
        self._km_distances = None
        self._regime_age = None
        self._ema_fast_arr = None
        self._ema_slow_arr = None
        self._ema_signal = None
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

        # K-Means clustering features
        from indicators.momentum.rsi import rsi
        rsi_arr = rsi(closes, period=14)
        features = np.column_stack([rsi_arr, self._atr, volumes])
        self._km_labels, self._km_distances = kmeans_regime(
            features, period=self.km_period, n_clusters=self.km_clusters)

        # Compute regime age (bars since last label change)
        self._regime_age = np.full(n, np.nan, dtype=np.float64)
        age = 0
        prev_label = np.nan
        for idx in range(n):
            if np.isnan(self._km_labels[idx]):
                continue
            if self._km_labels[idx] != prev_label:
                age = 0
                prev_label = self._km_labels[idx]
            else:
                age += 1
            self._regime_age[idx] = age

        # EMA cross
        self._ema_fast_arr, self._ema_slow_arr, self._ema_signal = ema_cross(
            closes, fast_period=self.ema_fast, slow_period=self.ema_slow)

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
        age_val = self._regime_age[i]
        ema_sig = self._ema_signal[i]
        dist_val = self._km_distances[i]
        if np.isnan(age_val) or np.isnan(ema_sig) or np.isnan(dist_val):
            return

        self.bars_since_last_scale += 1

        is_fresh = age_val <= self.regime_age_max
        # Use EMA position (fast vs slow) for ongoing direction
        ema_bullish = self._ema_fast_arr[i] > self._ema_slow_arr[i]
        ema_bearish = self._ema_fast_arr[i] < self._ema_slow_arr[i]

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

        # 3. Signal-based exit: EMA reversal or regime stale (>50 bars)
        if side == 1 and (ema_bearish or age_val > 50):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (ema_bullish or age_val > 50):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry: EMA cross in fresh regime
        if side == 0 and is_fresh:
            if ema_sig == 1.0:  # golden cross
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
            elif ema_sig == -1.0:  # death cross
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
            if self.direction == 1 and ema_bullish and is_fresh:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and ema_bearish and is_fresh:
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
