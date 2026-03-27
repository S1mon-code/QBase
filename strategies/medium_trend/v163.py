import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.pca_features import rolling_pca
from indicators.ml.ridge_forecast import rolling_ridge
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV163(TimeSeriesStrategy):
    """
    策略简介：PCA降维行情识别 + Ridge回归预测 + RSI确认的做多策略（日线）。

    使用指标：
    - Rolling PCA(60, 3): 从多特征提取主成分，第一主成分代表主趋势
    - Rolling Ridge(120): 用主成分预测未来收益方向
    - RSI(14): 避免追高，RSI<70时才入场
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Ridge预测 > 0（预测上涨）
    - PCA第一主成分方向为正（主趋势向上）
    - RSI < 70（非超买）

    出场条件：
    - ATR追踪止损 / 分层止盈
    - Ridge预测 < 0（预测下跌）

    优点：PCA降维去噪，Ridge正则化防过拟合，RSI防追高
    缺点：PCA主成分方向不稳定，可能在regime切换时失效
    """
    name = "mt_v163"
    warmup = 250
    freq = "daily"

    pca_period: int = 60            # Optuna: 40-100
    ridge_period: int = 120         # Optuna: 80-200
    rsi_upper: float = 70.0         # Optuna: 60-80
    atr_stop_mult: float = 3.0      # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._ridge_pred = None
        self._pca_comp = None
        self._rsi = None
        self._atr = None
        self._avg_volume = None

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
        self._rsi = rsi(closes, period=14)
        adx_arr = adx(highs, lows, closes, period=14)

        features = np.column_stack([self._rsi, adx_arr, self._atr])
        pca_comp, _ = rolling_pca(features, period=self.pca_period, n_components=3)
        self._pca_comp = pca_comp[:, 0]  # first principal component

        self._ridge_pred, _ = rolling_ridge(closes, features, period=self.ridge_period)
        self._avg_volume = fast_avg_volume(volumes, 20)

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
        pred = self._ridge_pred[i]
        pc1 = self._pca_comp[i]
        rsi_val = self._rsi[i]
        if np.isnan(pred) or np.isnan(pc1) or np.isnan(rsi_val):
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

        # 3. Signal exit
        if side == 1 and pred < 0:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and pred > 0 and pc1 > 0 and rsi_val < self.rsi_upper:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, pred):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, pred):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if pred <= 0:
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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
