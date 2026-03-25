import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.kalman_adaptive import adaptive_kalman
from indicators.regime.mean_variance_regime import mv_regime
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class KalmanPredictionErrorVarianceRegime(TimeSeriesStrategy):
    """
    Kalman Prediction Error + Variance Regime strategy for AG futures.

    Mean-reverts on Kalman filter prediction error in low-variance regime,
    trend-follows Kalman momentum in high-variance regime.

    Indicators:
    - Adaptive Kalman (period=60): trend, momentum, volatility estimate
    - Mean-Variance Regime (period=60): 0=low_vol, 1=normal, 2=high_vol
    - ATR(14): stop and profit targets

    Entry (Long - mean revert in low vol):
    - Variance regime = 0 (low vol)
    - Kalman prediction error negative (price below trend => oversold)

    Entry (Long - trend follow in high vol):
    - Variance regime = 2 (high vol)
    - Kalman momentum positive (trend direction up)

    Entry (Short): mirror logic

    Exit:
    - ATR trailing stop
    - Tiered profit-taking at 3ATR, 5ATR
    - Kalman momentum/error reversal

    Pros: Dual-mode adapts to market conditions; Kalman filter is optimal for linear systems
    Cons: Non-stationarity violates Kalman assumptions; regime classification lag
    """
    name = "v32_kalman_variance_regime"
    warmup = 60 * 3
    freq = "1h"

    kalman_period: int = 60
    mv_period: int = 60
    error_threshold: float = 0.5
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._kalman_trend = None
        self._kalman_momentum = None
        self._kalman_vol = None
        self._mv_label = None
        self._atr = None
        self._avg_volume = None
        self._prediction_error = None

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
        self._kalman_trend, self._kalman_momentum, self._kalman_vol = adaptive_kalman(
            closes, period=self.kalman_period
        )
        self._mv_label, _, _ = mv_regime(closes, period=self.mv_period, n_regimes=3)
        self._atr = atr(highs, lows, closes, period=14)

        # Prediction error: close - kalman_trend, normalized by kalman vol
        self._prediction_error = np.full(n, np.nan)
        for idx in range(n):
            if not np.isnan(self._kalman_trend[idx]) and not np.isnan(self._kalman_vol[idx]):
                kv = self._kalman_vol[idx]
                if kv > 1e-10:
                    self._prediction_error[idx] = (closes[idx] - self._kalman_trend[idx]) / kv

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

        pe = self._prediction_error[i]
        km = self._kalman_momentum[i]
        mv = self._mv_label[i]
        atr_val = self._atr[i]
        if np.isnan(pe) or np.isnan(km) or np.isnan(mv) or np.isnan(atr_val):
            return

        regime = int(mv)
        low_vol = regime == 0
        high_vol = regime == 2

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

        # Signal exit
        if side == 1 and ((low_vol and pe > self.error_threshold) or (high_vol and km < 0)):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and ((low_vol and pe < -self.error_threshold) or (high_vol and km > 0)):
            context.close_short()
            self._reset_state()
            return

        # Entry - Long (mean revert in low vol)
        if side == 0 and low_vol and pe < -self.error_threshold:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.lowest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # Entry - Short (mean revert in low vol)
        elif side == 0 and low_vol and pe > self.error_threshold:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.sell(base_lots)
                self.entry_price = price
                self.stop_price = price + self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.lowest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # Entry - Long (trend follow in high vol)
        elif side == 0 and high_vol and km > 0:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.lowest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # Entry - Short (trend follow in high vol)
        elif side == 0 and high_vol and km < 0:
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
        elif side == 1 and self._should_add_long(price, atr_val, True):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

        # Scaling - Short
        elif side == -1 and self._should_add_short(price, atr_val, True):
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
