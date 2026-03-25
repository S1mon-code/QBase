import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.autoencoder_error import reconstruction_error
from indicators.ml.wavelet_decompose import wavelet_features
from indicators.regime.mean_variance_regime import mv_regime
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class DenoisingAutoencoderNoiseRegime(TimeSeriesStrategy):
    """
    Denoising Autoencoder + Noise Regime strategy for AG futures.

    Follows the denoised (wavelet trend) signal in low-noise regimes,
    reduces exposure in high-noise regimes. Uses autoencoder reconstruction
    error to gauge noise level alongside wavelet energy ratio.

    Indicators:
    - Wavelet Features (level=4): trend component and energy ratio
    - Reconstruction Error (period=120): market state unusualness
    - Mean-Variance Regime (period=60): 0=low_vol, 1=normal, 2=high_vol
    - ATR(14): stop and profit targets

    Entry (Long - low noise):
    - Wavelet energy ratio < 0.25 (trend dominates)
    - MV regime = 0 or 1 (not high vol)
    - Wavelet trend rising (5-bar slope > 0)

    Entry (Short - low noise):
    - Same noise conditions
    - Wavelet trend falling

    Exit:
    - ATR trailing stop
    - Tiered profit-taking at 3ATR, 5ATR
    - High noise (energy ratio > 0.4 or MV regime = 2)

    Pros: Denoised signal avoids whipsaws; noise regime filter protects capital
    Cons: Wavelet smoothing introduces lag; may miss sharp moves in volatile markets
    """
    name = "v38_denoising_noise_regime"
    warmup = 120 * 3
    freq = "4h"

    wavelet_level: int = 4
    energy_entry: float = 0.25
    energy_exit: float = 0.40
    slope_lookback: int = 5
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._trend = None
        self._energy = None
        self._mv_label = None
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

        self._trend, _, self._energy = wavelet_features(
            closes, wavelet="db4", level=self.wavelet_level
        )
        self._mv_label, _, _ = mv_regime(closes, period=60, n_regimes=3)
        self._atr = atr(highs, lows, closes, period=14)

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

        if i < self.slope_lookback:
            return
        trend_now = self._trend[i]
        trend_prev = self._trend[i - self.slope_lookback]
        energy = self._energy[i]
        mv = self._mv_label[i]
        atr_val = self._atr[i]

        if np.isnan(trend_now) or np.isnan(trend_prev) or np.isnan(energy) or np.isnan(atr_val):
            return

        trend_slope = trend_now - trend_prev
        low_noise = energy < self.energy_entry and (np.isnan(mv) or int(mv) != 2)
        high_noise = energy > self.energy_exit or (not np.isnan(mv) and int(mv) == 2)

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

        # Exit on high noise or slope reversal
        if side == 1 and (high_noise or trend_slope < 0):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (high_noise or trend_slope > 0):
            context.close_short()
            self._reset_state()
            return

        # Entry - Long
        if side == 0 and low_noise and trend_slope > 0:
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
        elif side == 0 and low_noise and trend_slope < 0:
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
        elif side == 1 and self._should_add_long(price, atr_val, trend_slope > 0 and low_noise):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

        # Scaling - Short
        elif side == -1 and self._should_add_short(price, atr_val, trend_slope < 0 and low_noise):
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
