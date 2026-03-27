import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.ml.kalman_trend import kalman_filter
from indicators.regime.market_state import market_state
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV43(TimeSeriesStrategy):
    """
    策略简介：Kalman滤波趋势 + Market State行情识别的30min自适应策略。

    使用指标：
    - Kalman Filter: 平滑价格提取趋势方向，价格在Kalman线上方为多头
    - Market State(20): 行情状态识别，state=1为趋势状态时才交易
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - close > Kalman滤波值（价格在趋势线上方）
    - Market State = 趋势状态（非震荡环境）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - close < Kalman滤波值（价格跌破趋势线）

    优点：Kalman自适应噪声过滤，结合行情识别减少震荡假信号
    缺点：Market State分类可能滞后，Kalman参数对噪声敏感
    """
    name = "medium_trend_v43"
    warmup = 700
    freq = "30min"

    kalman_process_noise: float = 0.01   # Optuna: 0.001-0.1
    kalman_meas_noise: float = 1.0       # Optuna: 0.1-5.0
    ms_period: int = 20                  # Optuna: 10-40
    atr_stop_mult: float = 3.0           # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._kalman = None
        self._mstate = None
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
        oi = context.get_full_oi_array()

        self._kalman = kalman_filter(closes, process_noise=self.kalman_process_noise,
                                     measurement_noise=self.kalman_meas_noise)
        self._mstate = market_state(closes, volumes, oi, period=self.ms_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        kalman_val = self._kalman[i]
        ms_val = self._mstate[i]
        close_val = context.get_full_close_array()[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(kalman_val) or np.isnan(ms_val):
            return

        is_trending = (ms_val == 1)
        price_above_kalman = (close_val > kalman_val)

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

        # 3. Signal exit: price drops below Kalman
        if side == 1 and not price_above_kalman:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and price_above_kalman and is_trending:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, price_above_kalman, is_trending):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, above_kalman, is_trending):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if not above_kalman or not is_trending:
            return False
        return True

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
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
