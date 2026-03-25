import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.boosting_signal import gradient_boost_signal
from indicators.trend.supertrend import supertrend
from indicators.volume.oi_divergence import oi_divergence

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV158(TimeSeriesStrategy):
    """
    策略简介：Gradient Boosting信号 + Supertrend方向 + OI背离的多空策略。

    使用指标：
    - Gradient Boost Signal(120): GBT方向概率预测(0-1)，>0.5看多
    - Supertrend(10, 3.0): 趋势方向确认
    - OI Divergence(20): 持仓背离，辅助判断趋势持续性
    - ATR(14): 止损距离计算

    进场条件（做多）：GBT概率>0.6 + Supertrend=1 + OI无看空背离
    进场条件（做空）：GBT概率<0.4 + Supertrend=-1 + OI无看多背离

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Supertrend方向反转

    优点：GBT非线性捕获复杂模式，Supertrend确认方向
    缺点：GBT滚动训练计算量大，可能过拟合近期数据
    """
    name = "ag_alltime_v158"
    warmup = 400
    freq = "4h"

    gbt_bull_thresh: float = 0.6   # Optuna: 0.55-0.70
    gbt_bear_thresh: float = 0.4   # Optuna: 0.30-0.45
    st_period: int = 10            # Optuna: 7-20
    st_multiplier: float = 3.0    # Optuna: 2.0-5.0
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._avg_volume = None
        self._gbt_prob = None
        self._st_dir = None
        self._oi_div = None

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
        _, self._st_dir = supertrend(
            highs, lows, closes, period=self.st_period, multiplier=self.st_multiplier
        )
        self._oi_div = oi_divergence(closes, oi, period=20)

        # Build features for GBT
        from indicators.momentum.rsi import rsi
        from indicators.trend.adx import adx as adx_fn
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx_fn(highs, lows, closes, period=14)
        features = np.column_stack([rsi_arr, adx_arr, self._atr])

        self._gbt_prob, _ = gradient_boost_signal(
            closes, features, period=120, n_estimators=20
        )

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

        gbt_p = self._gbt_prob[i]
        st_dir = self._st_dir[i]
        oi_d = self._oi_div[i]
        if np.isnan(gbt_p) or np.isnan(st_dir):
            return
        if np.isnan(oi_d):
            oi_d = 0.0  # Default to neutral if OI data unavailable

        prev_st = self._st_dir[i - 1] if i > 0 else np.nan

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

        # 3. Signal exit: Supertrend flip
        if side == 1 and st_dir == -1:
            context.close_long()
            self._reset_state()
        elif side == -1 and st_dir == 1:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: GBT probability + Supertrend + OI divergence filter
        if side == 0:
            if gbt_p > self.gbt_bull_thresh and st_dir == 1 and oi_d >= -0.3:
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
            elif gbt_p < self.gbt_bear_thresh and st_dir == -1 and oi_d <= 0.3:
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
        elif side != 0 and self._should_add(price, atr_val, st_dir):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if self.direction == 1:
                    context.buy(add_lots)
                else:
                    context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, st_dir):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1:
            if price < self.entry_price + atr_val:
                return False
            if st_dir != 1:
                return False
        elif self.direction == -1:
            if price > self.entry_price - atr_val:
                return False
            if st_dir != -1:
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
