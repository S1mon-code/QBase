import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.spectral_clustering_regime import spectral_regime
from indicators.volume.klinger import klinger
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class SpectralKlingerStrategy(TimeSeriesStrategy):
    """
    策略简介：Spectral Clustering周期识别 + Klinger成交量确认的多空策略。

    使用指标：
    - Spectral Regime: 谱聚类识别市场周期状态，通过状态均值回报判断方向
    - Klinger Volume Oscillator: 成交量力量指标，KVO > Signal = 买压
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Spectral状态均值回报 > 0（当前周期利于做多）
    - KVO > Signal line（成交量确认买压）

    进场条件（做空）：
    - Spectral状态均值回报 < 0
    - KVO < Signal line（成交量确认卖压）

    出场条件：
    - ATR追踪止损 / 分层止盈 / KVO交叉反转

    优点：谱聚类捕捉非线性周期结构，Klinger结合量价趋势
    缺点：谱聚类计算开销大，Klinger在震荡市频繁交叉
    """
    name = "v54_spectral_klinger"
    warmup = 400
    freq = "daily"

    spectral_period: int = 120
    n_clusters: int = 3
    klinger_fast: int = 34
    klinger_slow: int = 55
    klinger_signal: int = 13
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._state_returns = None
        self._kvo = None
        self._kvo_signal = None
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
        n = len(closes)

        rsi_arr = rsi(closes, 14)
        atr_arr = atr(highs, lows, closes, 14)
        returns = np.full(n, np.nan)
        returns[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
        features = np.column_stack([rsi_arr, atr_arr / np.maximum(closes, 1e-9), returns])

        labels, _ = spectral_regime(features, period=self.spectral_period,
                                     n_clusters=self.n_clusters)

        # Pre-compute state mean returns
        self._state_returns = np.full(n, np.nan)
        lookback = 60
        for idx in range(lookback, n):
            label = labels[idx]
            if np.isnan(label):
                continue
            label = int(label)
            mask = labels[idx - lookback:idx] == label
            rets_window = returns[idx - lookback:idx]
            valid = mask & ~np.isnan(rets_window)
            if np.sum(valid) > 5:
                self._state_returns[idx] = np.mean(rets_window[valid])

        self._kvo, self._kvo_signal = klinger(
            highs, lows, closes, volumes,
            fast=self.klinger_fast, slow=self.klinger_slow, signal=self.klinger_signal)

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
        kvo_val = self._kvo[i]
        kvo_sig = self._kvo_signal[i]
        atr_val = self._atr[i]
        if np.isnan(state_ret) or np.isnan(kvo_val) or np.isnan(kvo_sig) or np.isnan(atr_val) or atr_val <= 0:
            return

        self.bars_since_last_scale += 1
        kvo_bull = kvo_val > kvo_sig
        kvo_bear = kvo_val < kvo_sig

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
        if side == 1 and kvo_bear:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and kvo_bull:
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场 ──
        if side == 0:
            if state_ret > 0 and kvo_bull:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
            elif state_ret < 0 and kvo_bear:
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
                    and state_ret > 0 and kvo_bull):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and state_ret < 0 and kvo_bear):
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
