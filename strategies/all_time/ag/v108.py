import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.bollinger import bollinger_width
from indicators.volume.klinger import klinger
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BBWidthKlingerStrategy(TimeSeriesStrategy):
    """
    策略简介：Bollinger Band Width收缩 + Klinger成交量方向的多空策略。

    使用指标：
    - BB Width(20, 2.0): BB宽度百分位，检测波动率收缩（squeeze前兆）
    - Klinger Volume Oscillator(34,55,13): 成交量力量方向
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - BB Width处于低位（< 20百分位，波动率收缩）
    - Klinger KVO > Signal line（成交量力量看多）

    进场条件（做空）：
    - BB Width处于低位（< 20百分位，波动率收缩）
    - Klinger KVO < Signal line（成交量力量看空）

    出场条件：
    - ATR追踪止损 / 分层止盈 / Klinger交叉反转

    优点：低波动率后的突破方向由Klinger预判，入场时机好
    缺点：波动率可能持续低迷不突破，Klinger在低量时噪音大
    """
    name = "v108_bbwidth_klinger"
    warmup = 300
    freq = "4h"

    bb_period: int = 20             # Optuna: 15-30
    bb_width_pctl: float = 0.20    # Optuna: 0.10-0.30
    klinger_fast: int = 34          # Optuna: 20-50
    klinger_slow: int = 55          # Optuna: 40-80
    klinger_signal: int = 13        # Optuna: 8-20
    atr_stop_mult: float = 3.0     # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._bb_width = None
        self._bb_width_pctl_arr = None
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
        self.direction = 0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        n = len(closes)

        self._bb_width = bollinger_width(closes, period=self.bb_period)
        # Compute rolling percentile of BB width
        lookback = 120
        self._bb_width_pctl_arr = np.full(n, np.nan)
        for idx in range(lookback, n):
            window = self._bb_width[idx - lookback:idx]
            valid = window[~np.isnan(window)]
            if len(valid) > 10:
                self._bb_width_pctl_arr[idx] = np.sum(valid < self._bb_width[idx]) / len(valid)

        self._kvo, self._kvo_signal = klinger(
            highs, lows, closes, volumes,
            fast=self.klinger_fast, slow=self.klinger_slow, signal=self.klinger_signal)
        self._atr = atr(highs, lows, closes, period=14)

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

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
        kvo = self._kvo[i]
        kvo_sig = self._kvo_signal[i]
        width_pctl = self._bb_width_pctl_arr[i]
        if np.isnan(kvo) or np.isnan(kvo_sig) or np.isnan(width_pctl):
            return

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

        # 3. Signal-based exit: Klinger cross reversal
        if side == 1 and kvo < kvo_sig:
            context.close_long()
            self._reset_state()
        elif side == -1 and kvo > kvo_sig:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: BB contraction + Klinger direction
        is_squeeze = width_pctl < self.bb_width_pctl
        if side == 0:
            if is_squeeze and kvo > kvo_sig:
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
            elif is_squeeze and kvo < kvo_sig:
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
            if self.direction == 1 and kvo > kvo_sig:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and kvo < kvo_sig:
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
