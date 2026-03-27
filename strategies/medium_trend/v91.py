import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.kalman_trend import kalman_filter
from indicators.momentum.rsi import rsi
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV91(TimeSeriesStrategy):
    """
    策略简介：Kalman Filter趋势估计 + RSI动量确认的4h多头策略。

    使用指标：
    - Kalman Filter: 自适应趋势线，价格>Kalman线为看多
    - RSI(14): > 50确认动量向上
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - 价格 > Kalman趋势线 且 Kalman线上升
    - RSI > 50

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - 价格 < Kalman趋势线（趋势破位）

    优点：Kalman自适应平滑，比固定周期均线更灵活
    缺点：噪声参数选择影响大
    """
    name = "medium_trend_v91"
    warmup = 200
    freq = "4h"

    kalman_process_noise: float = 0.01   # Optuna: 0.001-0.1
    kalman_measure_noise: float = 1.0    # Optuna: 0.5-5.0
    rsi_period: int = 14                 # Optuna: 10-20
    atr_stop_mult: float = 3.0           # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._kalman = None
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

        self._kalman = kalman_filter(
            closes, process_noise=self.kalman_process_noise,
            measurement_noise=self.kalman_measure_noise
        )
        self._rsi = rsi(closes, period=self.rsi_period)
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
        kf = self._kalman[i]
        rsi_val = self._rsi[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(kf) or np.isnan(rsi_val):
            return
        kf_prev = self._kalman[i - 1] if i > 0 else np.nan
        if np.isnan(kf_prev):
            return

        closes = context.get_full_close_array()
        close_price = closes[i]
        kf_rising = kf > kf_prev

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

        # 3. Signal exit: price below Kalman
        if side == 1 and close_price < kf:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and close_price > kf and kf_rising and rsi_val > 50:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, close_price, kf, rsi_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, close_price, kf, rsi_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if close_price <= kf or rsi_val < 50:
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
