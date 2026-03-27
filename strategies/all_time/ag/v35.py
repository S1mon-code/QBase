import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.online_regression import online_sgd_signal
from indicators.regime.regime_switch_speed import switch_speed
from indicators.regime.vol_regime_markov import vol_regime_simple
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class OnlineLearningRegimeSwitchSpeed(TimeSeriesStrategy):
    """
    Online Learning (SGD) + Regime Switch Speed strategy for AG futures.

    Uses an online SGD linear model that adapts its learning rate based on
    regime switching speed: faster adaptation when regimes switch frequently.

    Indicators:
    - Online SGD Signal (lr=0.01, period=20): incrementally updated linear model
    - Vol Regime Simple (period=60): 2-state regime for switch speed input
    - Switch Speed (period=60): average duration, switch frequency, current duration
    - ATR(14): stop and profit targets

    Entry (Long):
    - SGD signal > 0 (bullish prediction)
    - Weights norm > 0.1 (model has learned something meaningful)

    Entry (Short):
    - SGD signal < 0
    - Weights norm > 0.1

    Note: Learning rate is effectively modulated by regime speed:
    the strategy reduces position size when switching is fast (uncertain).

    Exit:
    - ATR trailing stop
    - Tiered profit-taking at 3ATR, 5ATR
    - Signal reversal

    Pros: Online learning adapts continuously; no retraining window
    Cons: SGD sensitive to learning rate; can overfit to noise in fast regime
    """
    name = "v35_online_learning_switch_speed"
    warmup = 60 * 3
    freq = "1h"

    sgd_lr: float = 0.01
    sgd_period: int = 20
    min_weights_norm: float = 0.1
    fast_switch_threshold: float = 0.3
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._sgd_signal = None
        self._weights_norm = None
        self._switch_freq = None
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

        self._sgd_signal, self._weights_norm = online_sgd_signal(
            closes, features, learning_rate=self.sgd_lr, period=self.sgd_period
        )

        vol_labels, _, _ = vol_regime_simple(closes, period=60)
        _, self._switch_freq, _ = switch_speed(vol_labels, period=60)
        self._atr = atr(highs, lows, closes, period=14)

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

        sig = self._sgd_signal[i]
        wn = self._weights_norm[i]
        sf = self._switch_freq[i]
        atr_val = self._atr[i]
        if np.isnan(sig) or np.isnan(atr_val):
            return
        if np.isnan(wn):
            wn = 0.0
        if np.isnan(sf):
            sf = 0.0

        has_signal = wn > self.min_weights_norm
        fast_switching = sf > self.fast_switch_threshold

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

        # Signal reversal exit
        if side == 1 and sig < 0:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and sig > 0:
            context.close_short()
            self._reset_state()
            return

        # Position size reduction factor when switching is fast
        size_factor = 0.5 if fast_switching else 1.0

        # Entry - Long
        if side == 0 and sig > 0 and has_signal:
            base_lots = self._calc_lots(context, atr_val)
            adjusted = max(1, int(base_lots * size_factor))
            if adjusted > 0:
                context.buy(adjusted)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.lowest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # Entry - Short
        elif side == 0 and sig < 0 and has_signal:
            base_lots = self._calc_lots(context, atr_val)
            adjusted = max(1, int(base_lots * size_factor))
            if adjusted > 0:
                context.sell(adjusted)
                self.entry_price = price
                self.stop_price = price + self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.lowest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # Scaling - Long
        elif side == 1 and self._should_add_long(price, atr_val, sig > 0):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

        # Scaling - Short
        elif side == -1 and self._should_add_short(price, atr_val, sig < 0):
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
