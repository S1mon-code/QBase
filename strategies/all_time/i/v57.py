import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.i.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.volatility.atr import atr
from indicators.trend.keltner import keltner
from indicators.momentum.rsi import rsi
from indicators.volume.volume_momentum import volume_momentum

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV57(TimeSeriesStrategy):
    """
    Keltner + RSI divergence proxy + Volume Momentum for Iron Ore (4h).

    Indicators:
    - Keltner Channel(20, 1.5): mean-reversion bands
    - RSI(14): when price makes new low but RSI doesn't = bullish divergence proxy
    - Volume Momentum(10): confirms participation

    Entry (Long): Price < lower KC + RSI higher than 5-bar-ago RSI (divergence) + vol mom > 0
    Entry (Short): Price > upper KC + RSI lower than 5-bar-ago RSI (divergence) + vol mom > 0
    Exit: Price returns to KC middle

    Pros: Divergence-based entries have higher conviction than pure oversold
    Cons: Divergence detection via lookback is approximate
    """
    name = "i_alltime_v57"
    warmup = 400
    freq = "4h"

    kc_period: int = 20
    kc_mult: float = 1.5
    rsi_period: int = 14
    div_lookback: int = 5
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._kc_upper = None
        self._kc_mid = None
        self._kc_lower = None
        self._rsi = None
        self._vol_mom = None
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

        self._kc_upper, self._kc_mid, self._kc_lower = keltner(highs, lows, closes, period=self.kc_period, mult=self.kc_mult)
        self._rsi = rsi(closes, period=self.rsi_period)
        self._vol_mom = volume_momentum(volumes, period=10)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        vol = context.volume
        if self._avg_volume[i] is not None and not np.isnan(self._avg_volume[i]):
            if vol < self._avg_volume[i] * 0.1:
                return

        kc_u = self._kc_upper[i]
        kc_m = self._kc_mid[i]
        kc_l = self._kc_lower[i]
        rsi_val = self._rsi[i]
        rsi_prev = self._rsi[max(0, i - self.div_lookback)]
        vm = self._vol_mom[i]
        c = context.get_full_close_array()[i]
        c_prev = context.get_full_close_array()[max(0, i - self.div_lookback)]
        atr_val = self._atr[i]
        if np.isnan(kc_u) or np.isnan(rsi_val) or np.isnan(rsi_prev) or np.isnan(vm) or np.isnan(atr_val):
            return

        self.bars_since_last_scale += 1

        # 1. Stop loss - Long
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return
        # 1. Stop loss - Short
        if side == -1:
            self.lowest_since_entry = min(self.lowest_since_entry, price)
            trailing = self.lowest_since_entry + self.atr_stop_mult * atr_val
            self.stop_price = min(self.stop_price, trailing)
            if price >= self.stop_price:
                context.close_short()
                self._reset_state()
                return

        # 2. Tiered profit - Long
        if side == 1 and self.entry_price > 0:
            profit_atr = (price - self.entry_price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_long(lots=max(1, lots // 3))
                self._took_profit_5atr = True
                self.position_scale = max(0, self.position_scale - 1)
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_long(lots=max(1, lots // 3))
                self._took_profit_3atr = True
                self.position_scale = max(0, self.position_scale - 1)
                return
        # 2. Tiered profit - Short
        if side == -1 and self.entry_price > 0:
            profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_5atr = True
                self.position_scale = max(0, self.position_scale - 1)
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_3atr = True
                self.position_scale = max(0, self.position_scale - 1)
                return

        # 3. Signal exit
        if side == 1 and c >= kc_m:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and c <= kc_m:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        bull_div = c < c_prev and rsi_val > rsi_prev  # price lower, RSI higher
        bear_div = c > c_prev and rsi_val < rsi_prev  # price higher, RSI lower
        if side == 0 and c < kc_l and bull_div and vm > 0:
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
        elif side == 0 and c > kc_u and bear_div and vm > 0:
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
        elif side == 1 and self._should_add_long(price, atr_val, rsi_val < 40):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self._should_add_short(price, atr_val, rsi_val > 60):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add_long(self, price, atr_val, signal_confirm):
        if self.position_scale >= MAX_SCALE or self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        return signal_confirm

    def _should_add_short(self, price, atr_val, signal_confirm):
        if self.position_scale >= MAX_SCALE or self.bars_since_last_scale < 10:
            return False
        if price > self.entry_price - atr_val:
            return False
        return signal_confirm

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_distance = self.atr_stop_mult * atr_val * spec.multiplier
        if stop_distance <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_distance)
        margin_per_lot = context.close_raw * spec.multiplier * spec.margin_rate
        if margin_per_lot <= 0:
            return 0
        max_lots = int(context.equity * 0.30 / margin_per_lot)
        return max(1, min(risk_lots, max_lots))

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
