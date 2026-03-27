import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.hmm_regime import hmm_regime
from indicators.regime.trend_persistence import trend_persistence
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class NeuralNetTrendPersistence(TimeSeriesStrategy):
    """
    Neural Net Signal (HMM proxy) + Trend Persistence strategy for AG futures.

    Uses HMM state detection as a neural-net-like signal combined with
    autocorrelation-based trend persistence to confirm entries.

    Indicators:
    - HMM Regime (3-state, period=252): identifies bullish/bearish/neutral states
    - Trend Persistence (max_lag=20, period=60): autocorrelation sum measuring
      trend strength; high values indicate persistent trending
    - ATR(14): stop distance and profit target calculation

    Entry (Long):
    - HMM state classified as bullish (highest mean return state)
    - Trend persistence above 60th percentile of its rolling history

    Entry (Short):
    - HMM state classified as bearish (lowest mean return state)
    - Trend persistence above 60th percentile of its rolling history

    Exit:
    - ATR trailing stop (highest/lowest since entry minus/plus N * ATR)
    - Tiered profit-taking at 3ATR and 5ATR
    - Signal reversal (HMM state flips)

    Pros: Dual confirmation reduces false signals; HMM captures regime shifts
    Cons: HMM retraining lag; trend persistence may miss early reversals
    """
    name = "v21_neural_net_trend_persistence"
    warmup = 252 * 3
    freq = "4h"

    hmm_period: int = 252
    persist_period: int = 60
    persist_pctl: float = 60.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._hmm_labels = None
        self._hmm_probs = None
        self._hmm_diag = None
        self._persistence = None
        self._dominant_lag = None
        self._atr = None
        self._avg_volume = None
        self._bullish_state = 0
        self._bearish_state = 0

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

        self._hmm_labels, self._hmm_probs, self._hmm_diag = hmm_regime(
            closes, n_states=3, period=self.hmm_period
        )
        self._persistence, self._dominant_lag = trend_persistence(
            closes, max_lag=20, period=self.persist_period
        )
        self._atr = atr(highs, lows, closes, period=14)

        # Identify bullish/bearish states by average return in each state
        n = len(closes)
        safe = np.maximum(closes, 1e-9)
        log_ret = np.full(n, np.nan)
        log_ret[1:] = np.log(safe[1:] / safe[:-1])
        state_returns = {}
        for i in range(n):
            lbl = self._hmm_labels[i]
            if not np.isnan(lbl) and not np.isnan(log_ret[i]):
                s = int(lbl)
                if s not in state_returns:
                    state_returns[s] = []
                state_returns[s].append(log_ret[i])
        avg_ret = {s: np.mean(r) for s, r in state_returns.items() if len(r) > 10}
        if len(avg_ret) >= 2:
            self._bullish_state = max(avg_ret, key=avg_ret.get)
            self._bearish_state = min(avg_ret, key=avg_ret.get)
        else:
            self._bullish_state = 0
            self._bearish_state = 2

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
        persist_val = self._persistence[i]
        atr_val = self._atr[i]
        if np.isnan(hmm_lbl) or np.isnan(persist_val) or np.isnan(atr_val):
            return

        # Rolling percentile of persistence
        lookback = 120
        start = max(0, i - lookback)
        hist = self._persistence[start:i + 1]
        valid_hist = hist[~np.isnan(hist)]
        if len(valid_hist) < 20:
            return
        pctl_thresh = np.percentile(valid_hist, self.persist_pctl)
        high_persist = persist_val >= pctl_thresh

        hmm_state = int(hmm_lbl)
        is_bullish = hmm_state == self._bullish_state
        is_bearish = hmm_state == self._bearish_state

        self.bars_since_last_scale += 1

        # Stop check - Long
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

        # Stop check - Short
        if side == -1:
            self.lowest_since_entry = min(self.lowest_since_entry, price)
            trailing = self.lowest_since_entry + self.atr_stop_mult * atr_val
            self.stop_price = min(self.stop_price, trailing)
            if price >= self.stop_price:
                context.close_short()
                self._reset_state()
                return

        # Profit taking - Long
        if side == 1 and self.entry_price > 0:
            profit_atr = (price - self.entry_price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                lots_to_close = max(1, lots // 3)
                context.close_long(lots=lots_to_close)
                self._took_profit_5atr = True
                self.position_scale = max(0, self.position_scale - 1)
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                lots_to_close = max(1, lots // 3)
                context.close_long(lots=lots_to_close)
                self._took_profit_3atr = True
                self.position_scale = max(0, self.position_scale - 1)
                return

        # Profit taking - Short
        if side == -1 and self.entry_price > 0:
            profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                lots_to_close = max(1, lots // 3)
                context.close_short(lots=lots_to_close)
                self._took_profit_5atr = True
                self.position_scale = max(0, self.position_scale - 1)
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                lots_to_close = max(1, lots // 3)
                context.close_short(lots=lots_to_close)
                self._took_profit_3atr = True
                self.position_scale = max(0, self.position_scale - 1)
                return

        # Signal reversal exit
        if side == 1 and is_bearish:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and is_bullish:
            context.close_short()
            self._reset_state()
            return

        # Entry - Long
        if side == 0 and is_bullish and high_persist:
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
        elif side == 0 and is_bearish and high_persist:
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

    def _should_add_long(self, price, atr_val, signal_confirm):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if not signal_confirm:
            return False
        return True

    def _should_add_short(self, price, atr_val, signal_confirm):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price > self.entry_price - atr_val:
            return False
        if not signal_confirm:
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
