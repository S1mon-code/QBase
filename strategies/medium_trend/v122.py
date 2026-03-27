import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.adx import adx
from indicators.momentum.stochastic import stochastic
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV122(TimeSeriesStrategy):
    """
    策略简介：4h ADX趋势确认 + 5min Stochastic超卖入场的多周期策略。

    使用指标：
    - ADX(14) [4h]: 趋势强度判断，> threshold确认趋势
    - Stochastic(%K, %D) [5min]: 超卖区域精确入场
    - ATR(14) [5min]: 止损距离计算

    进场条件（做多）：
    - 4h ADX > adx_threshold（趋势足够强）
    - 5min Stochastic %K < stoch_entry（超卖回调）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR / 5ATR）
    - 4h ADX 跌破阈值

    优点：ADX过滤震荡期，Stochastic精确入场
    缺点：ADX滞后，可能错过趋势启动初期
    """
    name = "medium_trend_v122"
    freq = "5min"
    warmup = 2000

    adx_period: int = 14
    adx_threshold: float = 25.0
    stoch_k: int = 14
    stoch_entry: float = 20.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._stoch_k = None
        self._atr = None
        self._avg_volume = None
        self._adx_4h = None
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

        k_arr, _ = stochastic(highs, lows, closes, k=self.stoch_k, d=3)
        self._stoch_k = k_arr
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 48  # 5min * 48 = 4h
        n_4h = n // step
        trim = n_4h * step
        closes_4h = closes[:trim].reshape(n_4h, step)[:, -1]
        highs_4h = highs[:trim].reshape(n_4h, step).max(axis=1)
        lows_4h = lows[:trim].reshape(n_4h, step).min(axis=1)

        self._adx_4h = adx(highs_4h, lows_4h, closes_4h, period=self.adx_period)
        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1),
                                  len(self._adx_4h) - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        stoch_val = self._stoch_k[i]
        atr_val = self._atr[i]
        adx_val = self._adx_4h[j]
        if np.isnan(stoch_val) or np.isnan(atr_val) or np.isnan(adx_val):
            return

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

        if side == 1 and adx_val < self.adx_threshold * 0.7:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and adx_val > self.adx_threshold and stoch_val < self.stoch_entry:
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
                    and adx_val > self.adx_threshold and stoch_val < 40):
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
