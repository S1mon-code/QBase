import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.linear_regression import linear_regression_slope, r_squared
from indicators.volume.force_index import force_index
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV93(TimeSeriesStrategy):
    """
    策略简介：线性回归斜率+R^2趋势质量 + Force Index量能确认的4h多头策略。

    使用指标：
    - Linear Regression Slope(20): 斜率 > 0确认上升趋势
    - R-Squared(20): > 0.5确认趋势一致性（线性度高）
    - Force Index(13): > 0确认多头力量
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - LR斜率 > 0 且 R^2 > r2_threshold
    - Force Index > 0

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - LR斜率 < 0（趋势反转）

    优点：R^2过滤低质量趋势，Force Index确认量能
    缺点：线性回归对突变不敏感
    """
    name = "medium_trend_v93"
    warmup = 200
    freq = "4h"

    lr_period: int = 20           # Optuna: 14-40
    r2_threshold: float = 0.5     # Optuna: 0.3-0.7
    force_period: int = 13        # Optuna: 8-20
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._lr_slope = None
        self._r2 = None
        self._force = None
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

        self._lr_slope = linear_regression_slope(closes, self.lr_period)
        self._r2 = r_squared(closes, self.lr_period)
        self._force = force_index(closes, volumes, period=self.force_period)
        self._atr = atr(highs, lows, closes, period=14)
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
        slope = self._lr_slope[i]
        r2_val = self._r2[i]
        fi = self._force[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(slope) or np.isnan(r2_val) or np.isnan(fi):
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

        # 3. Signal exit: slope reversal
        if side == 1 and slope < 0:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and slope > 0 and r2_val > self.r2_threshold and fi > 0:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, slope, r2_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, slope, r2_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if slope <= 0 or r2_val < self.r2_threshold:
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
