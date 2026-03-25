import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.adaptive_lms import lms_filter
from indicators.regime.momentum_regime import momentum_regime
from indicators.spread.relative_strength import relative_strength

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV131(TimeSeriesStrategy):
    """
    策略简介：LMS自适应滤波学习 + 动量状态识别 + 白银相对强度的多空策略。

    使用指标：
    - LMS Adaptive Filter(20, 0.01): 在线学习滤波，误差方向给出趋势信号
    - Momentum Regime(10, 60): 加速/减速/反转状态分类，只在加速状态交易
    - Relative Strength(20): 白银vs自身滚动均值的相对强度动量

    进场条件（做多）：LMS误差>0 + 动量加速(regime=1) + RS动量>0
    进场条件（做空）：LMS误差<0 + 动量加速(regime=1) + RS动量<0

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 动量反转(regime=-1)或LMS误差反向

    优点：自适应学习实时跟踪市场变化，动量状态过滤减少震荡假信号
    缺点：LMS学习率敏感，过大会过拟合噪声，过小会滞后
    """
    name = "ag_alltime_v131"
    warmup = 400
    freq = "4h"

    lms_period: int = 20           # Optuna: 10-40
    lms_mu: float = 0.01          # Optuna: 0.001-0.05
    mom_fast: int = 10            # Optuna: 5-20
    mom_slow: int = 60            # Optuna: 40-100
    rs_period: int = 20           # Optuna: 10-40
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._lms_error = None
        self._mom_regime = None
        self._rs_momentum = None
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

        # LMS adaptive filter — use error as directional signal
        _, self._lms_error, _ = lms_filter(closes, period=self.lms_period, mu=self.lms_mu)

        # Momentum regime
        self._mom_regime, _ = momentum_regime(closes, fast=self.mom_fast, slow=self.mom_slow)

        # Relative strength vs rolling mean (use closes as both asset and benchmark shifted)
        # Use smoothed close as benchmark
        from indicators.trend.ema import ema
        benchmark = ema(closes, period=60)
        _, self._rs_momentum, _ = relative_strength(closes, benchmark, period=self.rs_period)

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
        err_val = self._lms_error[i]
        regime_val = self._mom_regime[i]
        rs_mom = self._rs_momentum[i]
        if np.isnan(err_val) or np.isnan(regime_val) or np.isnan(rs_mom):
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

        # 3. Signal-based exit: momentum reversal or LMS error flip
        if side == 1 and (regime_val == -1 or err_val < 0):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (regime_val == -1 or err_val > 0):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        is_accelerating = regime_val == 1
        if side == 0 and is_accelerating:
            if err_val > 0 and rs_mom > 0:
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
            elif err_val < 0 and rs_mom < 0:
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
            if self.direction == 1 and err_val > 0 and is_accelerating:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and err_val < 0 and is_accelerating:
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
