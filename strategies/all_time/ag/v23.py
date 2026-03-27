import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.autoencoder_error import reconstruction_error
from indicators.regime.regime_score import composite_regime
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class AutoencoderAnomalyRegimeStability(TimeSeriesStrategy):
    """
    Autoencoder Anomaly + Regime Stability mean-reversion strategy for AG.

    Fades price moves when autoencoder reconstruction error spikes (anomaly)
    during stable/ranging regimes, expecting reversion to normal.

    Indicators:
    - Reconstruction Error (PCA-based, period=120): high error = unusual state
    - Composite Regime Score (period=20): identifies trending vs ranging
    - ATR(14): stop and profit targets

    Entry (Long - fade oversold anomaly):
    - Reconstruction error > 90th percentile (anomaly detected)
    - Regime is ranging (regime_score < -0.3)
    - Recent return is negative (price dropped => fade the drop)

    Entry (Short - fade overbought anomaly):
    - Reconstruction error > 90th percentile
    - Regime is ranging
    - Recent return is positive (price spiked => fade the spike)

    Exit:
    - ATR trailing stop
    - Tiered profit-taking at 3ATR, 5ATR
    - Error normalizes (drops below 50th percentile)

    Pros: Catches overreactions in stable markets
    Cons: Dangerous if regime shifts to trending during position
    """
    name = "v23_autoencoder_anomaly_regime"
    warmup = 120 * 3
    freq = "1h"

    ae_period: int = 120
    anomaly_pctl: float = 90.0
    normalize_pctl: float = 50.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._recon_error = None
        self._regime_score = None
        self._is_ranging = None
        self._atr = None
        self._avg_volume = None
        self._log_ret = None

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
        self._log_ret = np.full(n, np.nan)
        self._log_ret[1:] = np.log(safe[1:] / safe[:-1])

        # Build feature matrix for autoencoder
        feat_cols = []
        for lag in [1, 5, 10, 20]:
            f = np.full(n, np.nan)
            for idx in range(lag, n):
                f[idx] = (closes[idx] - closes[idx - lag]) / closes[idx - lag]
            feat_cols.append(f)
        vol20 = np.full(n, np.nan)
        for idx in range(20, n):
            vol20[idx] = np.std(self._log_ret[idx - 20:idx])
        feat_cols.append(vol20)
        features = np.column_stack(feat_cols)

        self._recon_error = reconstruction_error(features, period=self.ae_period, encoding_dim=2)
        self._regime_score, _, self._is_ranging = composite_regime(
            closes, highs, lows, period=20
        )
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

        error_val = self._recon_error[i]
        ranging = self._is_ranging[i]
        atr_val = self._atr[i]
        ret_val = self._log_ret[i]
        if np.isnan(error_val) or np.isnan(ranging) or np.isnan(atr_val) or np.isnan(ret_val):
            return

        # Rolling percentiles of error
        lookback = 120
        start = max(0, i - lookback)
        hist = self._recon_error[start:i + 1]
        valid_hist = hist[~np.isnan(hist)]
        if len(valid_hist) < 20:
            return
        anomaly_thresh = np.percentile(valid_hist, self.anomaly_pctl)
        normal_thresh = np.percentile(valid_hist, self.normalize_pctl)

        is_anomaly = error_val >= anomaly_thresh
        is_normal = error_val < normal_thresh
        is_ranging = ranging == 1.0

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

        # Exit when error normalizes
        if side == 1 and is_normal:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and is_normal:
            context.close_short()
            self._reset_state()
            return

        # Entry - Long (fade drop in ranging regime)
        if side == 0 and is_anomaly and is_ranging and ret_val < 0:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.lowest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # Entry - Short (fade spike in ranging regime)
        elif side == 0 and is_anomaly and is_ranging and ret_val > 0:
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
        elif side == 1 and self._should_add_long(price, atr_val, is_anomaly and is_ranging):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

        # Scaling - Short
        elif side == -1 and self._should_add_short(price, atr_val, is_anomaly and is_ranging):
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
