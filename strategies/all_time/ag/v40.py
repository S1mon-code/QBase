import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.hmm_regime import hmm_regime
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class HMMATRExpansion(TimeSeriesStrategy):
    """
    HMM + ATR Expansion strategy for AG futures.

    Enters on HMM state changes (regime transitions) when ATR is expanding,
    indicating increased volatility that often accompanies directional moves.

    Indicators:
    - HMM Regime (3-state, period=252): state labels and transition detection
    - ATR(14): stop distance, profit targets, and expansion detection
    - ATR Ratio: current ATR / 20-bar average ATR for expansion measurement

    Entry (Long):
    - HMM state just changed to bullish (highest avg return state)
    - ATR ratio > 1.3 (volatility expanding)

    Entry (Short):
    - HMM state just changed to bearish
    - ATR ratio > 1.3

    Exit:
    - ATR trailing stop
    - Tiered profit-taking at 3ATR, 5ATR
    - HMM state flips or ATR contracts (ratio < 0.8)

    Pros: State change + vol expansion catches breakouts; HMM identifies regimes
    Cons: HMM state labels can be noisy; ATR expansion may be temporary
    """
    name = "v40_hmm_atr_expansion"
    warmup = 252 * 3
    freq = "4h"

    hmm_period: int = 252
    atr_expansion: float = 1.3
    atr_contraction: float = 0.8
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._hmm_labels = None
        self._atr = None
        self._atr_ratio = None
        self._avg_volume = None
        self._bullish_state = 0
        self._bearish_state = 2

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
        self._hmm_labels, _, _ = hmm_regime(closes, n_states=3, period=self.hmm_period)
        self._atr = atr(highs, lows, closes, period=14)

        # ATR ratio: current / 20-bar average
        self._atr_ratio = np.full(n, np.nan)
        for idx in range(20, n):
            recent = self._atr[idx - 20:idx]
            valid = recent[~np.isnan(recent)]
            if len(valid) > 5 and not np.isnan(self._atr[idx]):
                avg = np.mean(valid)
                if avg > 1e-10:
                    self._atr_ratio[idx] = self._atr[idx] / avg

        # Identify bullish/bearish states
        safe = np.maximum(closes, 1e-9)
        log_ret = np.full(n, np.nan)
        log_ret[1:] = np.log(safe[1:] / safe[:-1])
        state_returns = {}
        for idx in range(n):
            lbl = self._hmm_labels[idx]
            if not np.isnan(lbl) and not np.isnan(log_ret[idx]):
                s = int(lbl)
                if s not in state_returns:
                    state_returns[s] = []
                state_returns[s].append(log_ret[idx])
        avg_ret = {s: np.mean(r) for s, r in state_returns.items() if len(r) > 10}
        if len(avg_ret) >= 2:
            self._bullish_state = max(avg_ret, key=avg_ret.get)
            self._bearish_state = min(avg_ret, key=avg_ret.get)

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
        vol = context.volume
        if self._avg_volume[i] is not None and not np.isnan(self._avg_volume[i]):
            if vol < self._avg_volume[i] * 0.1:
                return

        hmm_lbl = self._hmm_labels[i]
        atr_val = self._atr[i]
        atr_r = self._atr_ratio[i]
        if np.isnan(hmm_lbl) or np.isnan(atr_val) or np.isnan(atr_r):
            return

        hmm_state = int(hmm_lbl)
        is_bullish = hmm_state == self._bullish_state
        is_bearish = hmm_state == self._bearish_state

        # Detect state change
        state_changed = False
        if i > 0 and not np.isnan(self._hmm_labels[i - 1]):
            state_changed = hmm_state != int(self._hmm_labels[i - 1])

        expanding = atr_r > self.atr_expansion
        contracting = atr_r < self.atr_contraction

        self.bars_since_last_scale += 1

        # Stop - Long
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

        # Stop - Short
        if side == -1:
            self.lowest_since_entry = min(self.lowest_since_entry, price)
            trailing = self.lowest_since_entry + self.atr_stop_mult * atr_val
            self.stop_price = min(self.stop_price, trailing)
            if price >= self.stop_price:
                context.close_short()
                self._reset_state()
                return

        # Profit - Long
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

        # Profit - Short
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

        # Exit on HMM flip or ATR contraction
        if side == 1 and (is_bearish or contracting):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (is_bullish or contracting):
            context.close_short()
            self._reset_state()
            return

        # Entry - Long
        if side == 0 and state_changed and is_bullish and expanding:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.lowest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # Entry - Short
        elif side == 0 and state_changed and is_bearish and expanding:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.sell(base_lots)
                self.entry_price = price
                self.stop_price = price + self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.lowest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # Scaling - Long
        elif side == 1 and self._should_add_long(price, atr_val, is_bullish):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

        # Scaling - Short
        elif side == -1 and self._should_add_short(price, atr_val, is_bearish):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add_long(self, price, atr_val, confirm):
        if self.position_scale >= MAX_SCALE or self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val or not confirm:
            return False
        return True

    def _should_add_short(self, price, atr_val, confirm):
        if self.position_scale >= MAX_SCALE or self.bars_since_last_scale < 10:
            return False
        if price > self.entry_price - atr_val or not confirm:
            return False
        return True

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        from alphaforge.data.contract_specs import ContractSpecManager
        spec = ContractSpecManager().get(context.symbol)
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
