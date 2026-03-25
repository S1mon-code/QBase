import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.incremental_pca import incremental_pca_signal
from indicators.regime.trend_strength_composite import trend_strength
from indicators.momentum.roc import rate_of_change

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV139(TimeSeriesStrategy):
    """
    策略简介：增量PCA主成分方向 + 复合趋势强度过滤 + ROC突破的多空策略。

    使用指标：
    - Incremental PCA(3): 多特征降维，PC1方向作为市场主方向代理
    - Trend Strength(20): 复合趋势强度(0-100)，>50=有趋势
    - ROC(12): 变化率突破，>阈值做多，<-阈值做空

    进场条件（做多）：ROC>阈值 + PCA PC1方向>0 + 趋势强度>50
    进场条件（做空）：ROC<-阈值 + PCA PC1方向<0 + 趋势强度>50

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - ROC回中性或趋势强度<30

    优点：PCA降维提取多特征主方向，ROC简单高效捕捉动量爆发
    缺点：PCA降维可能丢失关键细节，ROC对噪声敏感需要过滤
    """
    name = "ag_alltime_v139"
    warmup = 400
    freq = "4h"

    pca_components: int = 3
    ts_period: int = 20            # Optuna: 10-40
    ts_thresh: float = 50.0        # Optuna: 30-70
    roc_period: int = 12           # Optuna: 8-20
    roc_thresh: float = 3.0        # Optuna: 1.0-5.0
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._pca_pc1 = None
        self._trend_str = None
        self._roc = None
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

        # PCA on feature matrix
        from indicators.momentum.rsi import rsi
        from indicators.trend.adx import adx
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx(highs, lows, closes, period=14)
        features = np.column_stack([rsi_arr, adx_arr, self._atr, volumes])
        pca_components, _ = incremental_pca_signal(features, n_components=self.pca_components)
        # PC1 = first principal component direction
        self._pca_pc1 = pca_components[:, 0]

        # Trend strength
        self._trend_str = trend_strength(closes, highs, lows, period=self.ts_period)

        # ROC
        self._roc = rate_of_change(closes, period=self.roc_period)

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
        pc1 = self._pca_pc1[i]
        ts_val = self._trend_str[i]
        roc_val = self._roc[i]
        if np.isnan(pc1) or np.isnan(ts_val) or np.isnan(roc_val):
            return

        self.bars_since_last_scale += 1

        strong_trend = ts_val > self.ts_thresh

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

        # 3. Signal-based exit: ROC neutral or trend weak
        if side == 1 and (roc_val < 0 or ts_val < 30):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (roc_val > 0 or ts_val < 30):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and strong_trend:
            if roc_val > self.roc_thresh and pc1 > 0:
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
            elif roc_val < -self.roc_thresh and pc1 < 0:
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
            if self.direction == 1 and roc_val > self.roc_thresh and strong_trend:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and roc_val < -self.roc_thresh and strong_trend:
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
