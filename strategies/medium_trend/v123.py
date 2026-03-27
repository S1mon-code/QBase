import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.hmm_regime import hmm_regime
from indicators.momentum.macd import macd
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV123(TimeSeriesStrategy):
    """
    策略简介：4h HMM regime识别 + 10min MACD交叉入场的多周期策略。

    使用指标：
    - HMM Regime(3 states) [4h]: 隐马尔可夫模型识别趋势/震荡/下跌状态
    - MACD(12,26,9) [10min]: 动量交叉入场信号
    - ATR(14) [10min]: 止损距离计算

    进场条件（做多）：
    - 4h HMM regime = 趋势状态（最高均值状态）
    - 10min MACD histogram > 0 且从负转正

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR / 5ATR）
    - HMM regime切换出趋势状态

    优点：HMM自适应识别市场状态，MACD捕捉动量转折
    缺点：HMM计算较重，regime标签可能不稳定
    """
    name = "medium_trend_v123"
    freq = "10min"
    warmup = 2000

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._macd_hist = None
        self._atr = None
        self._avg_volume = None
        self._hmm_4h = None
        self._hmm_bull_state = None
        self._4h_map = None

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
        n = len(closes)

        _, _, hist = macd(closes, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        self._macd_hist = hist
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 24  # 10min * 24 = 4h
        n_4h = n // step
        trim = n_4h * step
        closes_4h = closes[:trim].reshape(n_4h, step)[:, -1]

        self._hmm_4h = hmm_regime(closes_4h, n_states=3, period=252)
        # Identify bull state: the state with the highest mean return
        returns_4h = np.diff(closes_4h) / closes_4h[:-1]
        state_means = {}
        for s in range(3):
            mask = self._hmm_4h[1:] == s
            if mask.sum() > 0:
                state_means[s] = np.nanmean(returns_4h[mask])
            else:
                state_means[s] = -999.0
        self._hmm_bull_state = max(state_means, key=state_means.get)

        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1),
                                  len(self._hmm_4h) - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        macd_h = self._macd_hist[i]
        atr_val = self._atr[i]
        regime = self._hmm_4h[j]
        if np.isnan(macd_h) or np.isnan(atr_val) or np.isnan(regime):
            return

        prev_macd_h = self._macd_hist[i - 1] if i > 0 else np.nan
        is_bull = (regime == self._hmm_bull_state)

        self.bars_since_last_scale += 1

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

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

        if side == 1 and not is_bull:
            context.close_long()
            self._reset_state()
            return

        if (side == 0 and is_bull
                and not np.isnan(prev_macd_h) and prev_macd_h <= 0 and macd_h > 0):
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and is_bull and macd_h > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
