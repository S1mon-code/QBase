import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.hmm_regime import hmm_regime
from indicators.trend.supertrend import supertrend
from indicators.volume.cmf import cmf
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV195(TimeSeriesStrategy):
    """
    策略简介：HMM隐马尔可夫状态识别 + Supertrend方向 + CMF资金流的ML趋势策略。

    使用指标：
    - HMM Regime(3, 252): 3状态HMM识别市场regime，选择趋势状态交易
    - Supertrend(10, 3.0): 趋势方向确认
    - CMF(20): 资金流方向确认
    - ATR(14): 止损距离计算

    进场条件（做多）：HMM处于上涨regime + Supertrend=1 + CMF>0
    进场条件（做空）：HMM处于下跌regime + Supertrend=-1 + CMF<0

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - HMM状态切换或Supertrend翻转

    优点：HMM自适应识别市场状态，避免在震荡regime交易
    缺点：HMM重训练计算开销大，状态标签无固定含义需要后处理
    """
    name = "ag_alltime_v195"
    warmup = 300
    freq = "4h"

    hmm_states: int = 3           # Optuna: 2-4
    hmm_period: int = 252         # Optuna: 120-300
    st_period: int = 10           # Optuna: 7-15
    st_multiplier: float = 3.0   # Optuna: 2.0-5.0
    cmf_period: int = 20          # Optuna: 14-30
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._hmm_labels = None
        self._hmm_probs = None
        self._st_line = None
        self._st_dir = None
        self._cmf = None
        self._atr = None
        self._avg_volume = None
        self._bullish_state = None
        self._bearish_state = None

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

        self._hmm_labels, self._hmm_probs, _ = hmm_regime(
            closes, n_states=self.hmm_states, period=self.hmm_period)
        self._st_line, self._st_dir = supertrend(
            highs, lows, closes, self.st_period, self.st_multiplier)
        self._cmf = cmf(highs, lows, closes, volumes, self.cmf_period)
        self._atr = atr(highs, lows, closes, period=14)

        # Identify which HMM state is bullish/bearish by average return
        n = len(closes)
        rets = np.diff(closes) / closes[:-1]
        state_returns = {}
        for s in range(self.hmm_states):
            mask = self._hmm_labels[1:] == s
            valid = ~np.isnan(mask.astype(float))
            if np.any(mask & valid):
                state_returns[s] = np.nanmean(rets[mask])
            else:
                state_returns[s] = 0.0
        sorted_states = sorted(state_returns.items(), key=lambda x: x[1])
        self._bearish_state = sorted_states[0][0]
        self._bullish_state = sorted_states[-1][0]

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
        st_dir = self._st_dir[i]
        cmf_val = self._cmf[i]
        label = self._hmm_labels[i]

        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(st_dir) or np.isnan(cmf_val):
            return
        if np.isnan(label):
            return

        label = int(label)

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

        # 3. Signal exit: regime change or supertrend flip
        if side == 1 and (label == self._bearish_state or st_dir == -1):
            context.close_long()
            self._reset_state()
        elif side == -1 and (label == self._bullish_state or st_dir == 1):
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: bullish/bearish regime + supertrend + CMF
        if side == 0:
            if label == self._bullish_state and st_dir == 1 and cmf_val > 0:
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
            elif label == self._bearish_state and st_dir == -1 and cmf_val < 0:
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
            if self.direction == 1 and label == self._bullish_state and st_dir == 1:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and label == self._bearish_state and st_dir == -1:
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
