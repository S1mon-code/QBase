import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.gaussian_mixture_regime import gmm_regime
from indicators.structure.squeeze_detector import squeeze_probability
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class GaussianProcessSqueezeStrategy(TimeSeriesStrategy):
    """
    策略简介：GMM状态识别（Gaussian Process proxy）+ Squeeze释放入场。

    使用指标：
    - GMM Regime (Gaussian Process proxy): 高斯混合模型识别市场状态，
      通过状态转移方向判断趋势
    - Squeeze Probability: 空头/多头逼仓概率，释放时入场
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - GMM当前状态的均值收益 > 0（状态利于做多）
    - Short squeeze probability > 0.5（空头逼仓概率高=价格向上压力）

    进场条件（做空）：
    - GMM当前状态的均值收益 < 0
    - Long squeeze probability > 0.5（多头逼仓概率高=价格向下压力）

    出场条件：
    - ATR追踪止损 / 分层止盈 / 状态变化

    优点：GMM捕捉非线性状态转移，Squeeze是期货特有的强力信号
    缺点：GMM标签可能漂移，Squeeze误报在低OI时增加
    """
    name = "v53_gp_squeeze"
    warmup = 500
    freq = "4h"

    gmm_period: int = 120
    n_components: int = 3
    squeeze_period: int = 20
    squeeze_threshold: float = 0.5
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._gmm_labels = None
        self._state_returns = None
        self._ss_prob = None
        self._ls_prob = None
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

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()
        n = len(closes)

        rsi_arr = rsi(closes, 14)
        atr_arr = atr(highs, lows, closes, 14)
        returns = np.full(n, np.nan)
        returns[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
        features = np.column_stack([rsi_arr, atr_arr / np.maximum(closes, 1e-9), returns])

        self._gmm_labels, _ = gmm_regime(features, period=self.gmm_period,
                                           n_components=self.n_components)

        # Pre-compute state mean returns
        self._state_returns = np.full(n, np.nan)
        lookback = 60
        for idx in range(lookback, n):
            label = self._gmm_labels[idx]
            if np.isnan(label):
                continue
            label = int(label)
            mask = self._gmm_labels[idx - lookback:idx] == label
            rets_window = returns[idx - lookback:idx]
            valid = mask & ~np.isnan(rets_window)
            if np.sum(valid) > 5:
                self._state_returns[idx] = np.mean(rets_window[valid])

        self._ss_prob, self._ls_prob = squeeze_probability(
            closes, oi, volumes, period=self.squeeze_period)

        self._atr = atr(highs, lows, closes, period=self.atr_period)

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        state_ret = self._state_returns[i]
        ss_prob = self._ss_prob[i]
        ls_prob = self._ls_prob[i]
        atr_val = self._atr[i]
        if np.isnan(state_ret) or np.isnan(ss_prob) or np.isnan(ls_prob) or np.isnan(atr_val) or atr_val <= 0:
            return

        self.bars_since_last_scale += 1

        # ── 1. 止损 ──
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

        # ── 3. 信号反转退出 ──
        if side == 1 and state_ret < 0:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and state_ret > 0:
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场 ──
        if side == 0:
            if state_ret > 0 and ss_prob > self.squeeze_threshold:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
            elif state_ret < 0 and ls_prob > self.squeeze_threshold:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0

        # ── 5. 加仓 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and state_ret > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and state_ret < 0):
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
