import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.volatility.bollinger import bollinger_bands
from indicators.ml.pca_features import rolling_pca
from indicators.trend.adx import adx
from indicators.momentum.rsi import rsi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV159(TimeSeriesStrategy):
    """
    策略简介：PCA主成分方向 + ADX趋势确认 + BB宽度扩张的多空策略。

    使用指标：
    - Rolling PCA(60, 3): 多特征降维，第一主成分方向作为核心信号
    - ADX(14): 趋势强度，ADX>25确认趋势存在
    - Bollinger Bands(20, 2.0): BB宽度扩张确认波动率放大
    - ATR(14): 止损距离计算

    进场条件（做多）：PCA第一主成分>0 + ADX>25 + BB宽度在扩张
    进场条件（做空）：PCA第一主成分<0 + ADX>25 + BB宽度在扩张

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - PCA方向反转

    优点：PCA降维消除噪声特征，ADX+BB双重趋势确认
    缺点：PCA主成分含义不固定，日频信号较少
    """
    name = "ag_alltime_v159"
    warmup = 250
    freq = "daily"

    adx_threshold: float = 25.0   # Optuna: 18-35
    bb_width_lookback: int = 10   # Optuna: 5-20
    atr_stop_mult: float = 3.5    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._avg_volume = None
        self._pca_comp1 = None
        self._adx = None
        self._bb_width = None

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
        self._adx = adx(highs, lows, closes, period=14)

        bb_upper, bb_middle, bb_lower = bollinger_bands(closes, period=20, num_std=2.0)
        self._bb_width = np.full(n, np.nan)
        for idx in range(n):
            if not np.isnan(bb_upper[idx]) and not np.isnan(bb_lower[idx]) and bb_middle[idx] > 0:
                self._bb_width[idx] = (bb_upper[idx] - bb_lower[idx]) / bb_middle[idx]

        # Build features for PCA
        rsi_arr = rsi(closes, period=14)
        features = np.column_stack([rsi_arr, self._adx, self._atr, self._bb_width])

        pca_components, _ = rolling_pca(features, period=60, n_components=3)
        self._pca_comp1 = pca_components[:, 0]

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

        pca1 = self._pca_comp1[i]
        adx_val = self._adx[i]
        bb_w = self._bb_width[i]
        if np.isnan(pca1) or np.isnan(adx_val) or np.isnan(bb_w):
            return

        # Check BB width expanding
        lb = self.bb_width_lookback
        if i < lb:
            return
        prev_bb_w = self._bb_width[i - lb]
        if np.isnan(prev_bb_w):
            return
        bb_expanding = bb_w > prev_bb_w

        prev_pca1 = self._pca_comp1[i - 1] if i > 0 else np.nan
        if np.isnan(prev_pca1):
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

        # 3. Signal exit: PCA direction reversal
        if side == 1 and pca1 < 0 and prev_pca1 >= 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and pca1 > 0 and prev_pca1 <= 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: PCA direction + ADX + BB expanding
        if side == 0 and adx_val > self.adx_threshold and bb_expanding:
            if pca1 > 0:
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
            elif pca1 < 0:
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
        elif side != 0 and self._should_add(price, atr_val, pca1):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if self.direction == 1:
                    context.buy(add_lots)
                else:
                    context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, pca1):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1:
            if price < self.entry_price + atr_val:
                return False
            if pca1 <= 0:
                return False
        elif self.direction == -1:
            if price > self.entry_price - atr_val:
                return False
            if pca1 >= 0:
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
