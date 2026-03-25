import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.hmm_regime import hmm_regime
from indicators.regime.fractal_dimension import fractal_dim
from indicators.structure.squeeze_detector import squeeze_probability

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV135(TimeSeriesStrategy):
    """
    策略简介：HMM隐状态方向 + 分形维度趋势性 + 挤压释放突破的多空策略。

    使用指标：
    - HMM Regime(252, 3): 三状态隐马尔可夫模型，状态方向作为核心信号
    - Fractal Dimension(60): <1.4=趋势性，>1.6=随机，过滤趋势环境
    - Squeeze Probability(20): 空头/多头挤压概率，挤压释放时入场

    进场条件（做多）：HMM上涨状态 + 分形维<1.4(趋势) + 空头挤压概率>0.3
    进场条件（做空）：HMM下跌状态 + 分形维<1.4(趋势) + 多头挤压概率>0.3

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - HMM状态反转或分形维>1.6(失去趋势性)

    优点：HMM+分形维双重趋势确认，挤压释放提供高赔率入场点
    缺点：HMM重训练窗口大，挤压信号稀疏可能错过趋势起点
    """
    name = "ag_alltime_v135"
    warmup = 800
    freq = "4h"

    hmm_period: int = 252          # Optuna: 120-400
    hmm_states: int = 3
    fractal_period: int = 60       # Optuna: 30-100
    fractal_trend_thresh: float = 1.4  # Optuna: 1.3-1.5
    squeeze_period: int = 20       # Optuna: 10-40
    squeeze_thresh: float = 0.3    # Optuna: 0.2-0.5
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._hmm_labels = None
        self._hmm_probs = None
        self._fractal = None
        self._short_squeeze = None
        self._long_squeeze = None
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

        # HMM regime
        self._hmm_labels, self._hmm_probs, _ = hmm_regime(
            closes, n_states=self.hmm_states, period=self.hmm_period)

        # Fractal dimension
        self._fractal = fractal_dim(closes, period=self.fractal_period)

        # Squeeze detector
        self._short_squeeze, self._long_squeeze = squeeze_probability(
            closes, oi, volumes, period=self.squeeze_period)

        window = 20
        self._avg_volume = np.full_like(volumes, np.nan)
        for idx in range(window, len(volumes)):
            self._avg_volume[idx] = np.mean(volumes[idx - window:idx])

    def _get_hmm_direction(self, i):
        """Map HMM state label to direction. Determine by recent price trend in state."""
        label = self._hmm_labels[i]
        if np.isnan(label):
            return 0
        # Use state probabilities to determine direction
        probs = self._hmm_probs[i]
        if np.any(np.isnan(probs)):
            return 0
        # State with highest probability
        best_state = int(np.argmax(probs))
        # Simple heuristic: compare current and lagged state assignments
        # State 0 = first state, use recent returns to classify
        return best_state

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
        hmm_label = self._hmm_labels[i]
        frac_val = self._fractal[i]
        ss_val = self._short_squeeze[i]
        ls_val = self._long_squeeze[i]
        if np.isnan(hmm_label) or np.isnan(frac_val) or np.isnan(ss_val) or np.isnan(ls_val):
            return

        self.bars_since_last_scale += 1

        # Determine HMM direction from label
        # Labels are arbitrary; use state_probs to find trending states
        hmm_state = int(hmm_label)
        probs = self._hmm_probs[i]
        if np.any(np.isnan(probs)):
            return

        # Use simple heuristic: check recent 20-bar return to assign meaning to current state
        if i >= 20:
            closes_arr = context.get_full_close_array()
            recent_ret = closes_arr[i] / max(closes_arr[i - 20], 1e-9) - 1.0
        else:
            recent_ret = 0.0

        is_trending = frac_val < self.fractal_trend_thresh
        hmm_bullish = recent_ret > 0.01 and probs[hmm_state] > 0.4
        hmm_bearish = recent_ret < -0.01 and probs[hmm_state] > 0.4

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

        # 3. Signal-based exit: HMM flip or fractal goes random
        if side == 1 and (hmm_bearish or frac_val > 1.6):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (hmm_bullish or frac_val > 1.6):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry: HMM direction + trending fractal + squeeze release
        if side == 0 and is_trending:
            if hmm_bullish and ss_val > self.squeeze_thresh:
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
            elif hmm_bearish and ls_val > self.squeeze_thresh:
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
            if self.direction == 1 and hmm_bullish and is_trending:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and hmm_bearish and is_trending:
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
