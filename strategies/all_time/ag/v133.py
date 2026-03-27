import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.kalman_trend import kalman_filter
from indicators.regime.regime_score import composite_regime
from indicators.regime.oi_regime import oi_regime

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _simple_rl_signal(closes, atr_arr, period=120):
    """Simple Q-learning-like signal based on state-action reward tracking.

    Discretises market state into (trend_dir, vol_level) and tracks
    average reward for buy/sell actions. Returns signal array where
    positive = buy preferred, negative = sell preferred.
    """
    n = len(closes)
    signal = np.full(n, np.nan, dtype=np.float64)
    confidence = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return signal, confidence

    # Q-table: 2 trend dirs x 2 vol levels x 2 actions = 8 entries
    q_table = np.zeros((2, 2, 2), dtype=np.float64)
    counts = np.ones((2, 2, 2), dtype=np.float64)  # avoid div-by-zero

    log_ret = np.full(n, np.nan)
    log_ret[1:] = closes[1:] / np.maximum(closes[:-1], 1e-9) - 1.0

    for i in range(period, n):
        window_ret = log_ret[i - period + 1:i + 1]
        valid = window_ret[~np.isnan(window_ret)]
        if len(valid) < 20:
            continue

        trend_dir = 1 if np.mean(valid[-20:]) > 0 else 0
        vol_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 0
        rolling_atr = atr_arr[max(0, i - period):i + 1]
        valid_atr = rolling_atr[~np.isnan(rolling_atr)]
        vol_level = 1 if (len(valid_atr) > 0 and vol_val > np.median(valid_atr)) else 0

        # Update Q-table with recent reward
        if i > period and not np.isnan(log_ret[i]):
            reward = log_ret[i]
            # Action 0 = buy, action 1 = sell
            q_table[trend_dir, vol_level, 0] += (reward - q_table[trend_dir, vol_level, 0]) / counts[trend_dir, vol_level, 0]
            q_table[trend_dir, vol_level, 1] += (-reward - q_table[trend_dir, vol_level, 1]) / counts[trend_dir, vol_level, 1]
            counts[trend_dir, vol_level, 0] += 1
            counts[trend_dir, vol_level, 1] += 1

        q_buy = q_table[trend_dir, vol_level, 0]
        q_sell = q_table[trend_dir, vol_level, 1]
        signal[i] = q_buy - q_sell
        total_q = abs(q_buy) + abs(q_sell)
        confidence[i] = abs(q_buy - q_sell) / (total_q + 1e-9)

    return signal, confidence


class StrategyV133(TimeSeriesStrategy):
    """
    策略简介：简化强化学习信号 + 复合行情状态置信度 + OI Wyckoff阶段的多空策略。

    使用指标：
    - RL Signal(120): Q-learning风格的状态-动作奖励追踪，自适应学习最优方向
    - Composite Regime(20): 复合趋势/震荡评分，is_trending作为置信度过滤
    - OI Regime(60): Wyckoff OI阶段（积累/推升/分配/下跌），确认结构支持

    进场条件（做多）：RL信号>0 + 趋势状态(is_trending=1) + OI阶段=1(markup)
    进场条件（做空）：RL信号<0 + 趋势状态(is_trending=1) + OI阶段=3(markdown)

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - RL信号反向或OI阶段不支持

    优点：RL自动学习最优动作，OI结构确认减少假突破
    缺点：Q-table需要足够历史数据才能收敛，稀疏状态探索不足
    """
    name = "ag_alltime_v133"
    warmup = 500
    freq = "daily"

    rl_period: int = 120           # Optuna: 60-200
    regime_period: int = 20        # Optuna: 10-40
    oi_period: int = 60            # Optuna: 30-100
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._rl_signal = None
        self._rl_conf = None
        self._is_trending = None
        self._oi_phase = None
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

        # RL signal
        self._rl_signal, self._rl_conf = _simple_rl_signal(closes, self._atr, period=self.rl_period)

        # Composite regime
        _, self._is_trending, _ = composite_regime(closes, highs, lows, period=self.regime_period)

        # OI regime (Wyckoff phases)
        self._oi_phase, _ = oi_regime(closes, oi, volumes, period=self.oi_period)

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
        rl_sig = self._rl_signal[i]
        trending = self._is_trending[i]
        oi_phase = self._oi_phase[i]
        if np.isnan(rl_sig) or np.isnan(trending) or np.isnan(oi_phase):
            return

        self.bars_since_last_scale += 1

        is_trend = trending == 1
        oi_int = int(oi_phase)

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

        # 3. Signal-based exit
        if side == 1 and (rl_sig < 0 or oi_int == 3):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (rl_sig > 0 or oi_int == 1):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and is_trend:
            if rl_sig > 0 and oi_int == 1:  # markup
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
            elif rl_sig < 0 and oi_int == 3:  # markdown
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
            if self.direction == 1 and rl_sig > 0 and oi_int == 1:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and rl_sig < 0 and oi_int == 3:
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
