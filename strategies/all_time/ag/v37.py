import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.feature_importance import rolling_tree_importance
from indicators.ml.regime_persistence import regime_duration
from indicators.ml.hmm_regime import hmm_regime
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class FeatureImportanceRegimePersistence(TimeSeriesStrategy):
    """
    Feature Importance Shift + Regime Persistence strategy for AG futures.

    Anticipates regime changes by detecting shifts in which features matter
    most (feature importance instability) while regime persistence is declining.

    Indicators:
    - Rolling Tree Importance (period=120): RF-based feature importance per bar
    - HMM Regime (period=252): state labels for persistence tracking
    - Regime Duration: current duration and transition probability
    - ATR(14): stop and profit targets

    Entry (Long):
    - Feature importance shift detected (cosine similarity < 0.7 vs 10-bar-ago)
    - Regime transition prob rising (persistence declining)
    - Recent momentum positive (direction of anticipated change)

    Entry (Short):
    - Same importance shift + persistence decline
    - Recent momentum negative

    Exit:
    - ATR trailing stop
    - Tiered profit-taking at 3ATR, 5ATR
    - Momentum reversal

    Pros: Feature importance shift can signal regime changes before price does
    Cons: Importance calculation is noisy; false positives from random forest variance
    """
    name = "v37_feature_importance_regime_persist"
    warmup = 252 * 3
    freq = "daily"

    fi_period: int = 120
    hmm_period: int = 252
    similarity_threshold: float = 0.7
    shift_lookback: int = 10
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._importance = None
        self._trans_prob = None
        self._atr = None
        self._avg_volume = None
        self._mom5 = None

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
        safe = np.maximum(closes, 1e-9)
        log_ret = np.full(n, np.nan)
        log_ret[1:] = np.log(safe[1:] / safe[:-1])

        feat_cols = []
        for lag in [1, 5, 10, 20]:
            f = np.full(n, np.nan)
            for idx in range(lag, n):
                f[idx] = (closes[idx] - closes[idx - lag]) / closes[idx - lag]
            feat_cols.append(f)
        vol20 = np.full(n, np.nan)
        for idx in range(20, n):
            vol20[idx] = np.std(log_ret[idx - 20:idx])
        feat_cols.append(vol20)
        features = np.column_stack(feat_cols)

        self._importance = rolling_tree_importance(closes, features, period=self.fi_period)

        hmm_labels, _, _ = hmm_regime(closes, n_states=3, period=self.hmm_period)
        _, _, self._trans_prob = regime_duration(hmm_labels, period=60)

        self._atr = atr(highs, lows, closes, period=14)

        self._mom5 = np.full(n, np.nan)
        for idx in range(5, n):
            self._mom5[idx] = (closes[idx] - closes[idx - 5]) / closes[idx - 5]

        window = 20
        self._avg_volume = np.full_like(volumes, np.nan)
        for idx in range(window, len(volumes)):
            self._avg_volume[idx] = np.mean(volumes[idx - window:idx])

    def _cosine_similarity(self, a, b):
        """Cosine similarity between two vectors."""
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 1.0
        return dot / (na * nb)

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

        atr_val = self._atr[i]
        mom = self._mom5[i]
        tp = self._trans_prob[i]
        if np.isnan(atr_val) or np.isnan(mom):
            return

        # Check feature importance shift
        imp_now = self._importance[i]
        shifted = False
        if i >= self.shift_lookback and not np.any(np.isnan(imp_now)):
            imp_prev = self._importance[i - self.shift_lookback]
            if not np.any(np.isnan(imp_prev)):
                sim = self._cosine_similarity(imp_now, imp_prev)
                shifted = sim < self.similarity_threshold

        # Transition probability rising
        tp_rising = False
        if not np.isnan(tp) and i > 0:
            tp_prev = self._trans_prob[i - 1]
            if not np.isnan(tp_prev):
                tp_rising = tp > tp_prev and tp > 0.1

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

        # Signal exit: momentum reversal
        if side == 1 and mom < 0:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and mom > 0:
            context.close_short()
            self._reset_state()
            return

        # Entry - Long
        if side == 0 and shifted and tp_rising and mom > 0:
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
        elif side == 0 and shifted and tp_rising and mom < 0:
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
        elif side == 1 and self._should_add_long(price, atr_val, mom > 0):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

        # Scaling - Short
        elif side == -1 and self._should_add_short(price, atr_val, mom < 0):
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
