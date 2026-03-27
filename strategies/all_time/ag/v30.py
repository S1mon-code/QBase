import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.kmeans_regime import kmeans_regime
from indicators.ml.regime_persistence import regime_duration
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class KMeansCentroidShiftRegimeAge(TimeSeriesStrategy):
    """
    K-Means Centroid Shift + Regime Age strategy for AG futures.

    Detects centroid shifts (cluster label changes) and trades them
    only when the new regime is young (fresh transition).

    Indicators:
    - K-Means Regime (3 clusters, period=120): cluster labels and centroid distance
    - Regime Duration: current regime duration and average
    - ATR(14): stop and profit targets

    Entry (Long):
    - Cluster label just changed (centroid shift detected)
    - Current regime duration < 5 (very young regime)
    - Recent momentum positive (5-bar return > 0)

    Entry (Short):
    - Cluster label just changed
    - Current regime duration < 5
    - Recent momentum negative

    Exit:
    - ATR trailing stop
    - Tiered profit-taking at 3ATR, 5ATR
    - Regime ages beyond average duration or momentum reverses

    Pros: Catches regime transitions early; K-means adapts to market structure
    Cons: Cluster labels can be unstable; centroid retraining introduces noise
    """
    name = "v30_kmeans_centroid_regime_age"
    warmup = 120 * 3
    freq = "daily"

    kmeans_period: int = 120
    n_clusters: int = 3
    max_young_duration: int = 5
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._km_labels = None
        self._km_dist = None
        self._cur_duration = None
        self._avg_duration = None
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

        self._km_labels, self._km_dist = kmeans_regime(features, period=self.kmeans_period, n_clusters=self.n_clusters)
        self._cur_duration, self._avg_duration, _ = regime_duration(self._km_labels, period=60)
        self._atr = atr(highs, lows, closes, period=14)

        self._mom5 = np.full(n, np.nan)
        for idx in range(5, n):
            self._mom5[idx] = (closes[idx] - closes[idx - 5]) / closes[idx - 5]

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

        cur_dur = self._cur_duration[i]
        avg_dur = self._avg_duration[i]
        atr_val = self._atr[i]
        mom = self._mom5[i]
        lbl = self._km_labels[i]

        if np.isnan(atr_val) or np.isnan(mom) or np.isnan(cur_dur):
            return

        # Detect centroid shift (label change)
        label_changed = False
        if i > 0 and not np.isnan(lbl) and not np.isnan(self._km_labels[i - 1]):
            label_changed = lbl != self._km_labels[i - 1]

        young_regime = cur_dur <= self.max_young_duration
        old_regime = not np.isnan(avg_dur) and avg_dur > 0 and cur_dur > avg_dur

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

        # Exit on aged regime or momentum reversal
        if side == 1 and (old_regime or mom < 0):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (old_regime or mom > 0):
            context.close_short()
            self._reset_state()
            return

        # Entry - Long
        if side == 0 and label_changed and young_regime and mom > 0:
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
        elif side == 0 and label_changed and young_regime and mom < 0:
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
