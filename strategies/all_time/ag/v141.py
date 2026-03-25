import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.hmm_regime import hmm_regime
from indicators.ml.regime_transition_matrix import transition_features
from indicators.spread.cross_momentum import cross_momentum
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class LSTMRegimeTransitionCrossMomentumStrategy(TimeSeriesStrategy):
    """
    策略简介：HMM regime 在 regime 转换点 + 银/金跨品种动量确认的 4h 多空策略。

    使用指标：
    - HMM Regime(3 states, 252): 隐马尔可夫模型识别市场状态（代替 LSTM）
    - Transition Features: 检测 regime 转换概率，在转换窗口捕捉方向性变化
    - Cross Momentum(AG vs AU, 20): 银相对金的动量，银跑赢金=做多
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - HMM regime 刚发生转换（self_transition_prob < 0.5）
    - 当前 regime 为上升状态（regime == 1）
    - AG vs AU cross momentum > 0（银跑赢金）

    进场条件（做空）：
    - HMM regime 刚发生转换
    - 当前 regime 为下降状态（regime == 2）
    - AG vs AU cross momentum < 0

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Regime 再次转换退出

    优点：在 regime 转换初期捕捉趋势变化 + 跨品种验证减少假信号
    缺点：HMM 可能过拟合，regime 识别有滞后性
    """
    name = "v141_hmm_regime_transition_cross_momentum"
    warmup = 500
    freq = "4h"

    hmm_period: int = 252
    cm_period: int = 20
    atr_stop_mult: float = 3.0
    transition_threshold: float = 0.5

    def __init__(self):
        super().__init__()
        self._hmm_labels = None
        self._self_trans = None
        self._cross_mom = None
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

        # HMM regime detection
        self._hmm_labels = hmm_regime(closes, n_states=3, period=self.hmm_period)
        # Transition features from HMM labels
        self._self_trans, _, _ = transition_features(self._hmm_labels, n_states=3)
        self._atr = atr(highs, lows, closes, period=14)

        # Load AU auxiliary close for cross momentum
        au_closes = context.load_auxiliary_close("AU")
        if au_closes is not None and len(au_closes) == len(closes):
            self._cross_mom = cross_momentum(closes, au_closes, self.cm_period)
        else:
            self._cross_mom = np.zeros(len(closes))

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
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        regime = self._hmm_labels[i]
        self_trans = self._self_trans[i]
        cm_val = self._cross_mom[i]
        atr_val = self._atr[i]
        if np.isnan(regime) or np.isnan(atr_val):
            return
        if np.isnan(self_trans):
            self_trans = 1.0
        if np.isnan(cm_val):
            cm_val = 0.0

        self.bars_since_last_scale += 1
        regime = int(regime)
        is_transition = self_trans < self.transition_threshold

        # ── 1. 止损检查 ──
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

        # ── 2. 分层止盈 ──
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
        elif side == -1 and self.entry_price > 0:
            profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_3atr = True
                return

        # ── 3. Regime 再转换退出 ──
        if side == 1 and is_transition and regime != 1:
            context.close_long()
            self._reset_state()
        elif side == -1 and is_transition and regime != 2:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if is_transition and regime == 1 and cm_val > 0:
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
            elif is_transition and regime == 2 and cm_val < 0:
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

        # ── 5. 加仓逻辑 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and regime == 1 and cm_val > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and regime == 2 and cm_val < 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.sell(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
