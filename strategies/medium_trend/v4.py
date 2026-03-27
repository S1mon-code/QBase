import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.ml.hmm_regime import hmm_regime
from indicators.momentum.macd import macd
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV4(TimeSeriesStrategy):
    """
    策略简介：HMM行情状态识别 + MACD动量确认的做多策略（5min频率）。

    使用指标：
    - HMM Regime(3 states): 隐马尔可夫模型识别牛市/熊市/震荡状态
    - MACD(12,26,9): 动量方向确认
    - ATR(14): 止损距离计算

    进场条件（做多）：HMM判定为牛市状态 且 MACD柱状图 > 0
    出场条件：ATR追踪止损 / 分层止盈 / HMM状态切换 或 MACD转负

    优点：HMM捕捉隐含状态变化，比硬阈值更灵活
    缺点：HMM计算较慢，状态标签可能在不同训练窗口跳变
    """
    name = "mt_v4"
    warmup = 2000
    freq = "5min"

    hmm_period: int = 252
    hmm_states: int = 3
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._hmm = None
        self._macd_hist = None
        self._atr = None
        self._avg_volume = None
        self._bull_state = None

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

        self._hmm = hmm_regime(closes, n_states=self.hmm_states, period=self.hmm_period)
        macd_line, signal_line, hist = macd(closes, fast=self.macd_fast,
                                            slow=self.macd_slow, signal=self.macd_signal)
        self._macd_hist = hist
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        # Identify bull state: the HMM state with highest mean return
        n = len(closes)
        returns = np.zeros(n)
        returns[1:] = closes[1:] / closes[:-1] - 1
        state_means = {}
        for s in range(self.hmm_states):
            mask = self._hmm == s
            if np.any(mask):
                state_means[s] = np.nanmean(returns[mask])
        self._bull_state = max(state_means, key=state_means.get) if state_means else 0

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        hmm_state = self._hmm[i]
        macd_h = self._macd_hist[i]
        if np.isnan(atr_val) or np.isnan(hmm_state) or np.isnan(macd_h):
            return

        self.bars_since_last_scale += 1
        is_bull = int(hmm_state) == self._bull_state
        macd_positive = macd_h > 0

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

        # 3. Signal exit
        if side == 1 and (not is_bull or not macd_positive):
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and is_bull and macd_positive:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, is_bull, macd_positive):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, is_bull, macd_pos):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if not is_bull or not macd_pos:
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
