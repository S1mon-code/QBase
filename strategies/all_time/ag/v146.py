import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.volatility.bollinger import bollinger_bands
from indicators.ml.bayesian_trend import bayesian_online_trend
from indicators.regime.trend_persistence import trend_persistence

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV146(TimeSeriesStrategy):
    """
    策略简介：高斯过程(贝叶斯趋势)方向 + 趋势成熟度(早期) + BB %B回调入场的多空策略。

    使用指标：
    - Bayesian Trend(60): GP方向预测，正值看多、负值看空
    - Trend Persistence(60): 趋势成熟度判断，低值=早期趋势（更佳入场点）
    - Bollinger Bands(20, 2.0): %B回调信号，%B<0.2做多回调，%B>0.8做空回调
    - ATR(14): 止损距离计算

    进场条件（做多）：GP方向>0 + persistence<中位数(早期趋势) + BB %B<0.3
    进场条件（做空）：GP方向<0 + persistence<中位数(早期趋势) + BB %B>0.7

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - GP方向反转

    优点：GP概率方向+早期趋势入场，避免追高追低
    缺点：GP计算滞后，震荡市可能频繁翻转
    """
    name = "ag_alltime_v146"
    warmup = 300
    freq = "4h"

    bb_period: int = 20             # Optuna: 15-30
    bb_pctb_long: float = 0.3      # Optuna: 0.15-0.40
    bb_pctb_short: float = 0.7     # Optuna: 0.60-0.85
    atr_stop_mult: float = 3.0     # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._avg_volume = None
        self._bt_mean = None
        self._persist = None
        self._bb_upper = None
        self._bb_middle = None
        self._bb_lower = None

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
        _, _, self._bt_mean = bayesian_online_trend(closes, hazard_rate=0.01)
        self._persist, _ = trend_persistence(closes, max_lag=20, period=60)
        self._bb_upper, self._bb_middle, self._bb_lower = bollinger_bands(
            closes, period=self.bb_period, num_std=2.0
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
        bt_val = self._bt_mean[i]
        persist_val = self._persist[i]
        bb_u = self._bb_upper[i]
        bb_l = self._bb_lower[i]
        if np.isnan(bt_val) or np.isnan(persist_val) or np.isnan(bb_u) or np.isnan(bb_l):
            return

        bb_range = bb_u - bb_l
        pctb = (price - bb_l) / bb_range if bb_range > 0 else 0.5

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

        # 3. Signal-based exit: GP direction flip
        if side == 1 and bt_val < 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and bt_val > 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry
        persist_median = 0.5
        if side == 0:
            if bt_val > 0 and persist_val < persist_median and pctb < self.bb_pctb_long:
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
            elif bt_val < 0 and persist_val < persist_median and pctb > self.bb_pctb_short:
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
        elif side != 0 and self._should_add(price, atr_val, bt_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if self.direction == 1:
                    context.buy(add_lots)
                else:
                    context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, bt_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1:
            if price < self.entry_price + atr_val:
                return False
            if bt_val <= 0:
                return False
        elif self.direction == -1:
            if price > self.entry_price - atr_val:
                return False
            if bt_val >= 0:
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
