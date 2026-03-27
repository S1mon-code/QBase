import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.boosting_signal import gradient_boost_signal
from indicators.regime.variance_ratio import variance_ratio_test
from indicators.structure.position_crowding import position_crowding

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV138(TimeSeriesStrategy):
    """
    策略简介：梯度提升方向预测 + 方差比率动量状态 + 持仓拥挤度缩减的多空策略。

    使用指标：
    - Gradient Boost Signal(120): 滚动GBT分类器预测下一bar涨跌概率
    - Variance Ratio Test(60, 5): VR>1=趋势/动量态，VR<1=均值回归态
    - Position Crowding(60): 拥挤度高时缩减仓位，降低踩踏风险

    进场条件（做多）：Boost概率>0.6 + VR z-score>1(动量态)
    进场条件（做空）：Boost概率<0.4 + VR z-score>1(动量态)

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Boost概率回中性(0.45-0.55)或VR z-score<0

    优点：GBT自适应学习非线性模式，VR过滤确保在动量环境下交易
    缺点：GBT重训练频繁开销大，拥挤信号滞后
    """
    name = "ag_alltime_v138"
    warmup = 500
    freq = "daily"

    boost_period: int = 120        # Optuna: 60-200
    vr_period: int = 60            # Optuna: 30-100
    vr_holding: int = 5            # Optuna: 3-10
    crowd_period: int = 60         # Optuna: 30-100
    crowd_thresh: float = 0.7      # Optuna: 0.5-0.9
    boost_long_thresh: float = 0.6  # Optuna: 0.55-0.70
    boost_short_thresh: float = 0.4  # Optuna: 0.30-0.45
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._boost_signal = None
        self._vr = None
        self._vr_zscore = None
        self._crowding = None
        self._unwind_risk = None
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
        oi = context.get_full_oi_array()

        self._atr = atr(highs, lows, closes, period=14)

        # Gradient boosting signal
        from indicators.momentum.rsi import rsi
        from indicators.trend.adx import adx
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx(highs, lows, closes, period=14)
        features = np.column_stack([rsi_arr, adx_arr, self._atr, volumes])
        self._boost_signal, _ = gradient_boost_signal(
            closes, features, period=self.boost_period)

        # Variance ratio test
        self._vr, self._vr_zscore = variance_ratio_test(
            closes, period=self.vr_period, holding=self.vr_holding)

        # Position crowding
        self._crowding, self._unwind_risk = position_crowding(
            closes, oi, volumes, period=self.crowd_period)

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
        boost_val = self._boost_signal[i]
        vr_z = self._vr_zscore[i]
        crowd_val = self._crowding[i]
        if np.isnan(boost_val) or np.isnan(vr_z) or np.isnan(crowd_val):
            return

        self.bars_since_last_scale += 1

        is_momentum = vr_z > 1.0
        is_crowded = crowd_val > self.crowd_thresh

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

        # 2.5. Crowding scale-down: reduce position when crowded
        if side != 0 and is_crowded and lots > 1:
            reduce = max(1, lots // 3)
            if side == 1:
                context.close_long(lots=reduce)
            else:
                context.close_short(lots=reduce)
            return

        # 3. Signal-based exit: boost neutral or VR reversal
        if side == 1 and (boost_val < 0.45 or vr_z < 0):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (boost_val > 0.55 or vr_z < 0):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and is_momentum and not is_crowded:
            if boost_val > self.boost_long_thresh:
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
            elif boost_val < self.boost_short_thresh:
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

        # 5. Scale-in (skip if crowded)
        elif side != 0 and self._should_add(price, atr_val) and not is_crowded:
            if self.direction == 1 and boost_val > self.boost_long_thresh and is_momentum:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and boost_val < self.boost_short_thresh and is_momentum:
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
