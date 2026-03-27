import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.ml.hmm_regime import hmm_regime
from indicators.trend.donchian import donchian
from indicators.momentum.rsi import rsi
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV148(TimeSeriesStrategy):
    """
    策略简介：日线HMM regime + 4h Donchian突破 + 5min RSI入场的三周期策略。

    使用指标：
    - HMM(3 states) [日线]: 隐马尔可夫识别趋势状态
    - Donchian(20) [4h]: 中周期通道突破确认
    - RSI(14) [5min]: 超卖回调入场
    - ATR(14) [5min]: 止损距离

    进场条件（做多）：日线HMM=bull + 4h close>Donchian mid + 5min RSI<35
    出场条件：ATR追踪止损, 分层止盈, HMM切出bull

    优点：HMM自适应识别市场regime，Donchian简洁
    缺点：HMM计算较重，标签可能抖动
    """
    name = "medium_trend_v148"
    freq = "5min"
    warmup = 3000

    rsi_entry: float = 35.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._rsi = None
        self._atr = None
        self._avg_volume = None
        self._hmm_d = None
        self._bull_state = None
        self._dc_mid_4h = None
        self._closes_4h = None
        self._d_map = None
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

        self._rsi = rsi(closes, 14)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step_4h = 48
        n_4h = n // step_4h
        trim_4h = n_4h * step_4h
        closes_4h = closes[:trim_4h].reshape(n_4h, step_4h)[:, -1]
        highs_4h = highs[:trim_4h].reshape(n_4h, step_4h).max(axis=1)
        lows_4h = lows[:trim_4h].reshape(n_4h, step_4h).min(axis=1)

        _, _, self._dc_mid_4h = donchian(highs_4h, lows_4h, period=20)
        self._closes_4h = closes_4h
        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step_4h - 1), n_4h - 1)

        # Daily = 4h for Chinese futures
        self._hmm_d = hmm_regime(closes_4h, n_states=3, period=252)
        returns_d = np.diff(closes_4h) / closes_4h[:-1]
        state_means = {}
        for s in range(3):
            mask = self._hmm_d[1:] == s
            state_means[s] = np.nanmean(returns_d[mask]) if mask.sum() > 0 else -999.0
        self._bull_state = max(state_means, key=state_means.get)
        self._d_map = self._4h_map

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        rsi_val = self._rsi[i]
        atr_val = self._atr[i]
        regime = self._hmm_d[j]
        dc_mid = self._dc_mid_4h[j]
        c4h = self._closes_4h[j]
        if np.isnan(rsi_val) or np.isnan(atr_val) or np.isnan(regime) or np.isnan(dc_mid):
            return

        is_bull = (regime == self._bull_state)
        above_mid = c4h > dc_mid
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

        if side == 0 and is_bull and above_mid and rsi_val < self.rsi_entry:
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
                    and is_bull and above_mid):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
