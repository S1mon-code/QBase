import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.hmm_regime import hmm_regime
from indicators.regime.fractal_dimension import fractal_dim
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class HMMFractalDimension(TimeSeriesStrategy):
    """
    HMM + Fractal Dimension strategy for AG futures.

    Goes long when HMM detects bullish state and fractal dimension is low
    (smooth trend), short when bearish with low fractal dimension.

    Indicators:
    - HMM Regime (3-state, period=252): bullish/bearish/neutral classification
    - Fractal Dimension (period=60): Higuchi estimator; ~1.0=smooth, ~1.5=random
    - ATR(14): stop and profit targets

    Entry (Long):
    - HMM state = bullish (highest avg return state)
    - Fractal dim < 1.35 (smooth, trending behavior)

    Entry (Short):
    - HMM state = bearish (lowest avg return state)
    - Fractal dim < 1.35

    Exit:
    - ATR trailing stop
    - Tiered profit-taking at 3ATR, 5ATR
    - Fractal dim rises > 1.5 (random walk = no edge) or HMM flips

    Pros: Low fractal dim confirms real trend vs noise; HMM captures regimes
    Cons: Fractal dim can lag; HMM retraining introduces look-ahead risk
    """
    name = "v28_hmm_fractal_dim"
    warmup = 252 * 3
    freq = "4h"

    hmm_period: int = 252
    fractal_period: int = 60
    fractal_entry: float = 1.35
    fractal_exit: float = 1.50
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._hmm_labels = None
        self._fractal = None
        self._atr = None
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
        self._fractal = fractal_dim(closes, period=self.fractal_period)
        self._atr = atr(highs, lows, closes, period=14)

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
        self._avg_volume = fast_avg_volume(volumes, window)

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
        fd = self._fractal[i]
        atr_val = self._atr[i]
        if np.isnan(hmm_lbl) or np.isnan(fd) or np.isnan(atr_val):
            return

        is_bullish = int(hmm_lbl) == self._bullish_state
        is_bearish = int(hmm_lbl) == self._bearish_state
        low_fd = fd < self.fractal_entry
        high_fd = fd > self.fractal_exit

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

        # Exit on high fractal dim or HMM flip
        if side == 1 and (high_fd or is_bearish):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (high_fd or is_bullish):
            context.close_short()
            self._reset_state()
            return

        # Entry - Long
        if side == 0 and is_bullish and low_fd:
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
        elif side == 0 and is_bearish and low_fd:
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
        elif side == 1 and self._should_add_long(price, atr_val, is_bullish and low_fd):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

        # Scaling - Short
        elif side == -1 and self._should_add_short(price, atr_val, is_bearish and low_fd):
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
