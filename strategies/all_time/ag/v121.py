import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.feature_importance import rolling_tree_importance
from indicators.regime.trend_persistence import trend_persistence
from indicators.momentum.macd import macd

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV121(TimeSeriesStrategy):
    """
    策略简介：Random Forest 特征重要性信号 + 趋势持续性过滤 + MACD 动量确认的多空策略。

    使用指标：
    - Rolling Tree Importance(120): RF 特征重要性评估信号稳定性
    - Trend Persistence(60): 高持续性过滤震荡市假信号
    - MACD(12,26,9): 动量方向确认 + 直方图强度

    进场条件（做多）：趋势持续性 > 阈值，MACD histogram > 0，RF 重要性排名前列
    进场条件（做空）：趋势持续性 > 阈值，MACD histogram < 0，RF 重要性排名前列

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - MACD histogram 反向

    优点：RF 动态评估特征有效性，趋势持续性减少噪音
    缺点：RF 回测窗口较长，对快速反转反应迟钝
    """
    name = "ag_alltime_v121"
    warmup = 400
    freq = "4h"

    rf_period: int = 120
    persist_period: int = 60
    persist_thresh: float = 0.5
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._importance = None
        self._persist = None
        self._macd_hist = None
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

        # Build features for RF importance
        from indicators.momentum.rsi import rsi
        from indicators.trend.adx import adx as adx_fn
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx_fn(highs, lows, closes, period=14)
        atr_arr = self._atr.copy()
        features = np.column_stack([rsi_arr, adx_arr, atr_arr])
        self._importance = rolling_tree_importance(closes, features, period=self.rf_period, n_estimators=50)

        self._persist = trend_persistence(closes, max_lag=20, period=self.persist_period)
        _, _, self._macd_hist = macd(closes, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)

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
        persist_val = self._persist[i]
        hist_val = self._macd_hist[i]
        if np.isnan(persist_val) or np.isnan(hist_val):
            return

        # Check importance - use mean of feature importances as signal quality
        imp_row = self._importance[i]
        if np.any(np.isnan(imp_row)):
            imp_ok = False
        else:
            imp_ok = np.max(imp_row) > 0.2  # at least one strong feature

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
        if side == 1 and hist_val < 0:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and hist_val > 0:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and persist_val > self.persist_thresh and imp_ok:
            if hist_val > 0:
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
            elif hist_val < 0:
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
            if self.direction == 1 and hist_val > 0:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and hist_val < 0:
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
