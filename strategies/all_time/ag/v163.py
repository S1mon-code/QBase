import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.nearest_neighbor_signal import knn_signal
from indicators.trend.hma import hma
from indicators.seasonality.seasonal_momentum import seasonal_momentum
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class SVMHMASeasonalStrategy(TimeSeriesStrategy):
    """
    策略简介：SVM代理（KNN信号）+ HMA趋势方向 + 季节性动量对齐的日频多空策略。

    使用指标：
    - KNN Signal(120,5): 替代SVM的非参数ML预测方向
    - HMA(20): Hull均线趋势方向，价格>HMA=多头
    - Seasonal Momentum(3yr): 历史同期回报率，正=看多季节
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - KNN预测 > 0（ML看多）
    - 价格 > HMA（趋势向上）
    - 季节动量 > 0（有利季节）

    进场条件（做空）：
    - KNN预测 < 0（ML看空）
    - 价格 < HMA（趋势向下）
    - 季节动量 < 0（不利季节）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - HMA方向反转

    优点：ML+趋势+季节性三维确认，季节性提供独特alpha
    缺点：KNN在高维特征空间维数灾难，季节性规律可能失效
    """
    name = "v163_svm_hma_seasonal"
    warmup = 400
    freq = "daily"

    hma_period: int = 20
    knn_period: int = 120
    knn_k: int = 5
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._knn_pred = None
        self._hma = None
        self._seasonal = None
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
        self._hma = hma(closes, self.hma_period)

        # ML features
        rets = np.zeros_like(closes)
        rets[1:] = closes[1:] / closes[:-1] - 1
        vol20 = np.full_like(closes, np.nan)
        for idx in range(20, len(closes)):
            vol20[idx] = np.std(rets[idx-20:idx])
        features = np.column_stack([rets, vol20])
        features = np.nan_to_num(features, nan=0.0)
        self._knn_pred, _ = knn_signal(closes, features, period=self.knn_period, k=self.knn_k)

        # Seasonal momentum
        datetimes = context.get_full_datetime_array()
        self._seasonal, _ = seasonal_momentum(closes, datetimes, lookback_years=3)

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

        knn_val = self._knn_pred[i]
        hma_val = self._hma[i]
        seas_val = self._seasonal[i]
        atr_val = self._atr[i]
        if np.isnan(knn_val) or np.isnan(hma_val) or np.isnan(atr_val):
            return
        if np.isnan(seas_val):
            seas_val = 0.0

        self.bars_since_last_scale += 1

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

        # ── 3. HMA方向反转退出 ──
        if side == 1 and price < hma_val:
            context.close_long()
            self._reset_state()
        elif side == -1 and price > hma_val:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if knn_val > 0 and price > hma_val and seas_val > 0:
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
            elif knn_val < 0 and price < hma_val and seas_val < 0:
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
                    and knn_val > 0 and price > hma_val):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and knn_val < 0 and price < hma_val):
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
