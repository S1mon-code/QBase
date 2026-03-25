import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.kmeans_regime import kmeans_regime
from indicators.momentum.macd import macd
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV156(TimeSeriesStrategy):
    """
    策略简介：K-Means聚类(趋势cluster) + MACD交叉 + ATR动态仓位的多空策略。

    使用指标：
    - K-Means Regime(120, 3): 聚类识别趋势/震荡/突破cluster
    - MACD(12, 26, 9): 金叉死叉方向信号
    - ADX(14): 趋势强度确认，在趋势cluster中ADX>25
    - ATR(14): 止损距离计算 + 动态仓位调整

    进场条件（做多）：处于趋势cluster + MACD金叉(histogram>0) + ADX>25
    进场条件（做空）：处于趋势cluster + MACD死叉(histogram<0) + ADX>25

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - MACD histogram反转

    优点：K-Means自适应识别市场状态，MACD信号清晰
    缺点：聚类标签不稳定，MACD在震荡市多假信号
    """
    name = "ag_alltime_v156"
    warmup = 400
    freq = "4h"

    adx_threshold: float = 25.0   # Optuna: 18-35
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._avg_volume = None
        self._km_labels = None
        self._km_distances = None
        self._macd_hist = None
        self._adx = None
        self._trending_cluster = None

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
        self._adx = adx(highs, lows, closes, period=14)
        _, _, self._macd_hist = macd(closes, fast=12, slow=26, signal=9)

        # Build features for K-Means
        rsi_arr = rsi(closes, period=14)
        features = np.column_stack([rsi_arr, self._adx, self._atr])
        self._km_labels, self._km_distances = kmeans_regime(features, period=120, n_clusters=3)

        # Identify trending cluster: cluster with highest average ADX
        n = len(closes)
        self._trending_cluster = np.full(n, np.nan)
        for idx in range(120, n):
            window_labels = self._km_labels[idx - 120:idx]
            window_adx = self._adx[idx - 120:idx]
            best_cluster = -1
            best_avg_adx = -1
            for c in range(3):
                mask = window_labels == c
                valid_adx = window_adx[mask]
                valid_adx = valid_adx[~np.isnan(valid_adx)]
                if len(valid_adx) > 5:
                    avg_adx = np.mean(valid_adx)
                    if avg_adx > best_avg_adx:
                        best_avg_adx = avg_adx
                        best_cluster = c
            self._trending_cluster[idx] = best_cluster

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

        km_label = self._km_labels[i]
        adx_val = self._adx[i]
        macd_h = self._macd_hist[i]
        trend_cluster = self._trending_cluster[i]
        if np.isnan(km_label) or np.isnan(adx_val) or np.isnan(macd_h) or np.isnan(trend_cluster):
            return

        in_trend_cluster = (km_label == trend_cluster)
        prev_macd_h = self._macd_hist[i - 1] if i > 0 else np.nan

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

        # 3. Signal exit: MACD histogram reversal
        if not np.isnan(prev_macd_h):
            if side == 1 and macd_h < 0 and prev_macd_h >= 0:
                context.close_long()
                self._reset_state()
            elif side == -1 and macd_h > 0 and prev_macd_h <= 0:
                context.close_short()
                self._reset_state()

        side, lots = context.position

        # 4. Entry: in trending cluster + MACD crossover + ADX
        if side == 0 and in_trend_cluster and adx_val > self.adx_threshold:
            if not np.isnan(prev_macd_h):
                if macd_h > 0 and prev_macd_h <= 0:
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
                elif macd_h < 0 and prev_macd_h >= 0:
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
        elif side != 0 and self._should_add(price, atr_val, macd_h):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if self.direction == 1:
                    context.buy(add_lots)
                else:
                    context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, macd_h):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1:
            if price < self.entry_price + atr_val:
                return False
            if macd_h <= 0:
                return False
        elif self.direction == -1:
            if price > self.entry_price - atr_val:
                return False
            if macd_h >= 0:
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
