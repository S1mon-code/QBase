import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.kalman_trend import kalman_filter
from indicators.regime.variance_ratio import variance_ratio_test

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV117(TimeSeriesStrategy):
    """
    策略简介：Kalman滤波趋势方向 + Variance Ratio动量/均值回归判断 + ATR止损。

    使用指标：
    - Kalman Filter(0.01, 1.0): 自适应趋势提取，slope>0=上涨
    - Variance Ratio Test(60, 5): VR>1=动量(趋势)，VR<1=均值回归
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Kalman slope > 0（滤波趋势向上）
    - Variance Ratio z-score > 1.0（确认动量regime）
    进场条件（做空）：
    - Kalman slope < 0（滤波趋势向下）
    - Variance Ratio z-score > 1.0（确认动量regime）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Kalman slope反转或VR z-score回落

    优点：Kalman无延迟趋势估计，VR统计检验确保趋势有效性
    缺点：Kalman对噪声参数敏感，VR在短窗口可能不稳定
    """
    name = "ag_alltime_v117"
    warmup = 300
    freq = "1h"

    kf_process: float = 0.01      # Optuna: 0.001-0.1
    kf_measure: float = 1.0       # Optuna: 0.1-5.0
    vr_period: int = 60           # Optuna: 30-120
    vr_holding: int = 5           # Optuna: 3-10
    vr_z_thresh: float = 1.0      # Optuna: 0.5-2.0
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._kf_level = None
        self._kf_slope = None
        self._vr = None
        self._vr_z = None
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
        self._kf_level, self._kf_slope, _ = kalman_filter(
            closes, process_noise=self.kf_process,
            measurement_noise=self.kf_measure,
        )
        self._vr, self._vr_z = variance_ratio_test(
            closes, period=self.vr_period, holding=self.vr_holding
        )

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
        slope = self._kf_slope[i]
        vr_z = self._vr_z[i]
        if np.isnan(slope) or np.isnan(vr_z):
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

        # 3. Signal-based exit: slope reversal or VR drops
        if side == 1 and (slope < 0 or vr_z < 0):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (slope > 0 or vr_z < 0):
            context.close_short()
            self._reset_state()
            return

        # Re-read position
        side, lots = context.position

        # 4. Entry: Kalman direction + momentum regime
        if side == 0 and vr_z > self.vr_z_thresh:
            if slope > 0:
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
            elif slope < 0:
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
            signal_ok = (self.direction == 1 and slope > 0 and vr_z > 0.5) or \
                        (self.direction == -1 and slope < 0 and vr_z > 0.5)
            if signal_ok:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    if self.direction == 1:
                        context.buy(add_lots)
                    else:
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
