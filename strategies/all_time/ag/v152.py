import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.online_regression import online_sgd_signal
from indicators.regime.regime_switch_speed import switch_speed
from indicators.regime.market_state import market_state
from indicators.trend.hma import hma

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV152(TimeSeriesStrategy):
    """
    策略简介：在线学习(SGD) + regime切换速度 + HMA趋势跟踪的多空策略。

    使用指标：
    - Online SGD Signal: 在线增量学习，快速适应市场变化
    - Switch Speed(60): regime切换频率，高切换=适合快速模型
    - HMA(20): Hull均线趋势方向，HMA上升做多、下降做空
    - ATR(14): 止损距离计算

    进场条件（做多）：SGD信号>0 + 切换速度高(switch_freq>中位数) + HMA上升
    进场条件（做空）：SGD信号<0 + 切换速度高 + HMA下降

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - HMA方向反转

    优点：在线学习快速适应，HMA低延迟
    缺点：SGD对噪声敏感，1h频率交易成本高
    """
    name = "ag_alltime_v152"
    warmup = 500
    freq = "1h"

    hma_period: int = 20           # Optuna: 10-40
    atr_stop_mult: float = 2.5    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._avg_volume = None
        self._sgd_signal = None
        self._switch_freq = None
        self._hma = None

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
        self._hma = hma(closes, period=self.hma_period)

        # Build features for online SGD
        from indicators.momentum.rsi import rsi
        rsi_arr = rsi(closes, period=14)
        atr_arr = self._atr
        features = np.column_stack([rsi_arr, atr_arr])

        self._sgd_signal, _ = online_sgd_signal(
            closes, features, learning_rate=0.01, period=20
        )

        # Use market_state for regime labels, then compute switch speed
        ms_labels, _ = market_state(closes, volumes, oi, period=20)
        _, self._switch_freq, _ = switch_speed(ms_labels, period=60)

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

        sgd_sig = self._sgd_signal[i]
        sw_freq = self._switch_freq[i]
        hma_val = self._hma[i]
        if np.isnan(sgd_sig) or np.isnan(sw_freq) or np.isnan(hma_val):
            return
        if i < 1 or np.isnan(self._hma[i - 1]):
            return

        hma_dir = 1 if hma_val > self._hma[i - 1] else -1

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

        # 3. Signal exit: HMA direction flip
        if side == 1 and hma_dir == -1:
            context.close_long()
            self._reset_state()
        elif side == -1 and hma_dir == 1:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: SGD + high switch speed + HMA direction
        if side == 0 and sw_freq > 0.1:
            if sgd_sig > 0 and hma_dir == 1:
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
            elif sgd_sig < 0 and hma_dir == -1:
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
        elif side != 0 and self._should_add(price, atr_val, hma_dir):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if self.direction == 1:
                    context.buy(add_lots)
                else:
                    context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, hma_dir):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1:
            if price < self.entry_price + atr_val:
                return False
            if hma_dir != 1:
                return False
        elif self.direction == -1:
            if price > self.entry_price - atr_val:
                return False
            if hma_dir != -1:
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
