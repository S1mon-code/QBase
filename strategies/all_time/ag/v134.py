import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.feature_importance import rolling_tree_importance
from indicators.volatility.price_acceleration import price_acceleration
from indicators.momentum.cci import cci

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV134(TimeSeriesStrategy):
    """
    策略简介：随机森林特征重要性稳定性 + 价格加速度趋势 + CCI方向的多空策略。

    使用指标：
    - Rolling Tree Importance(120): 特征重要性稳定性，重要性集中=市场结构清晰
    - Price Acceleration(14): 二阶导数，加速=趋势增强
    - CCI(20): 商品通道指数方向，>100做多，<-100做空

    进场条件（做多）：特征重要性集中(top1>0.4) + 正加速 + CCI>100
    进场条件（做空）：特征重要性集中(top1>0.4) + 负加速 + CCI<-100

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - CCI回归中性区间或加速度反转

    优点：特征重要性集中度作为市场可预测性代理，加速度捕捉趋势强化
    缺点：随机森林重训练成本高，特征重要性随窗口可能不稳定
    """
    name = "ag_alltime_v134"
    warmup = 500
    freq = "4h"

    fi_period: int = 120           # Optuna: 60-200
    accel_period: int = 14         # Optuna: 10-20
    cci_period: int = 20           # Optuna: 14-30
    cci_thresh: float = 100.0      # Optuna: 50-150
    importance_thresh: float = 0.4  # Optuna: 0.3-0.6
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._fi_max = None
        self._accel = None
        self._is_accel = None
        self._cci = None
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

        # Feature importance — build feature matrix from basic indicators
        from indicators.momentum.rsi import rsi
        from indicators.trend.adx import adx
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx(highs, lows, closes, period=14)
        features = np.column_stack([rsi_arr, adx_arr, self._atr, volumes])
        importance_matrix = rolling_tree_importance(closes, features, period=self.fi_period)
        # Max importance across features — higher = more concentrated/predictable
        self._fi_max = np.nanmax(importance_matrix, axis=1)

        # Price acceleration
        self._accel, _, self._is_accel = price_acceleration(closes, period=self.accel_period)

        # CCI
        self._cci = cci(highs, lows, closes, period=self.cci_period)

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
        fi_val = self._fi_max[i]
        accel_val = self._accel[i]
        cci_val = self._cci[i]
        if np.isnan(fi_val) or np.isnan(accel_val) or np.isnan(cci_val):
            return

        self.bars_since_last_scale += 1

        features_stable = fi_val > self.importance_thresh

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

        # 3. Signal-based exit: CCI back to neutral or acceleration reversal
        if side == 1 and (cci_val < 0 or accel_val < 0):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (cci_val > 0 or accel_val > 0):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and features_stable:
            if cci_val > self.cci_thresh and accel_val > 0:
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
            elif cci_val < -self.cci_thresh and accel_val < 0:
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
            if self.direction == 1 and cci_val > self.cci_thresh and features_stable:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and cci_val < -self.cci_thresh and features_stable:
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
