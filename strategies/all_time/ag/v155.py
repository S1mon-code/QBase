import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.hmm_regime import hmm_regime
from indicators.momentum.rsi import rsi
from indicators.volume.volume_spike import volume_spike

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV155(TimeSeriesStrategy):
    """
    策略简介：HMM状态转换 + RSI背离 + 成交量放大的多空策略。

    使用指标：
    - HMM Regime(3, 252): 隐马尔可夫状态检测，捕捉regime转换
    - RSI(14): 背离信号，价格新高但RSI未创新高（或反向）
    - Volume Spike(20, 2.0): 成交量放大确认突破有效性
    - ATR(14): 止损距离计算

    进场条件（做多）：HMM转入上升state + RSI>50且未超买 + 成交量放大
    进场条件（做空）：HMM转入下降state + RSI<50且未超卖 + 成交量放大

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - HMM state再次转换

    优点：HMM概率模型捕捉非线性regime转换
    缺点：HMM训练不稳定，state标签可能漂移
    """
    name = "ag_alltime_v155"
    warmup = 400
    freq = "4h"

    rsi_period: int = 14           # Optuna: 10-20
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._avg_volume = None
        self._hmm_labels = None
        self._hmm_probs = None
        self._hmm_persist = None
        self._rsi = None
        self._vol_spike = None

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
        self._prev_hmm_state = np.nan

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._atr = atr(highs, lows, closes, period=14)
        self._hmm_labels, self._hmm_probs, self._hmm_persist = hmm_regime(
            closes, n_states=3, period=252
        )
        self._rsi = rsi(closes, period=self.rsi_period)
        self._vol_spike = volume_spike(volumes, period=20, threshold=2.0)

        # Compute average return per HMM state to assign meaning
        n = len(closes)
        self._state_return_sign = np.full(n, np.nan)
        returns = np.full(n, np.nan)
        returns[1:] = np.diff(closes) / closes[:-1]

        for idx in range(252, n):
            window_labels = self._hmm_labels[idx - 252:idx]
            window_returns = returns[idx - 252:idx]
            current_state = self._hmm_labels[idx]
            if np.isnan(current_state):
                continue
            mask = window_labels == current_state
            valid = window_returns[mask]
            valid = valid[~np.isnan(valid)]
            if len(valid) > 5:
                self._state_return_sign[idx] = np.sign(np.mean(valid))

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

        hmm_state = self._hmm_labels[i]
        rsi_val = self._rsi[i]
        state_sign = self._state_return_sign[i]
        if np.isnan(hmm_state) or np.isnan(rsi_val) or np.isnan(state_sign):
            return

        prev_state = self._hmm_labels[i - 1] if i > 0 else np.nan
        state_changed = (not np.isnan(prev_state)) and (hmm_state != prev_state)
        has_vol_spike = bool(self._vol_spike[i]) if i < len(self._vol_spike) else False

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

        # 3. Signal exit: HMM state switch against position
        if side == 1 and state_changed and state_sign < 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and state_changed and state_sign > 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: HMM transition + RSI confirmation + volume spike
        if side == 0 and state_changed and has_vol_spike:
            if state_sign > 0 and rsi_val > 40 and rsi_val < 70:
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
            elif state_sign < 0 and rsi_val < 60 and rsi_val > 30:
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
        elif side != 0 and self._should_add(price, atr_val, state_sign):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if self.direction == 1:
                    context.buy(add_lots)
                else:
                    context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, state_sign):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1:
            if price < self.entry_price + atr_val:
                return False
            if state_sign <= 0:
                return False
        elif self.direction == -1:
            if price > self.entry_price - atr_val:
                return False
            if state_sign >= 0:
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
        self._prev_hmm_state = np.nan
