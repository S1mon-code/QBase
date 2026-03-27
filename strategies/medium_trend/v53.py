import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.keltner import keltner
from indicators.momentum.stochastic import stochastic
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV53(TimeSeriesStrategy):
    """
    策略简介：Keltner Channel突破 + Stochastic回调入场的30min通道回调策略。

    使用指标：
    - Keltner Channel(20, 10, 1.5): 价格在上轨上方确认多头趋势
    - Stochastic(14, 3): %K < 30时为趋势中超卖回调
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - close > Keltner上轨（强势确认）
    - Stochastic %K < stoch_entry（超卖回调入场）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - close < Keltner中轨

    优点：在突破基础上等待回调，进场价位更优
    缺点：强势突破后可能不回调，导致错过信号
    """
    name = "medium_trend_v53"
    warmup = 600
    freq = "30min"

    kc_ema: int = 20
    kc_atr: int = 10
    kc_mult: float = 1.5
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_entry: float = 30.0         # Optuna: 20-40
    atr_stop_mult: float = 3.0       # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._kc_upper = None
        self._kc_mid = None
        self._stoch_k = None
        self._atr = None
        self._avg_volume = None
        self._was_above_upper = False

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self._was_above_upper = False

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._kc_upper, self._kc_mid, _ = keltner(highs, lows, closes,
                                                     ema=self.kc_ema, atr=self.kc_atr, mult=self.kc_mult)
        self._stoch_k, _ = stochastic(highs, lows, closes, k=self.stoch_k, d=self.stoch_d)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        close_val = context.get_full_close_array()[i]
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        kc_u = self._kc_upper[i]
        kc_m = self._kc_mid[i]
        sk = self._stoch_k[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(kc_u) or np.isnan(sk):
            return

        # Track if price was recently above upper band
        if close_val > kc_u:
            self._was_above_upper = True
        elif close_val < kc_m:
            self._was_above_upper = False

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

        if side == 1 and close_val < kc_m:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and self._was_above_upper and sk < self.stoch_entry and close_val > kc_m:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, close_val, kc_m):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, close_val, kc_m):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if close_val < kc_m:
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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self._was_above_upper = False
