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
from indicators.momentum.rsi import rsi
from indicators.volume.obv import obv

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _kalman_filter(prices, process_noise=1e-3, measurement_noise=1e-1):
    """Simple 1D Kalman filter returning smoothed price and slope."""
    n = len(prices)
    smoothed = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    if n < 2:
        return smoothed, slope
    x = float(prices[0])
    p = 1.0
    dx = 0.0
    for i in range(n):
        x += dx
        p += process_noise
        k = p / (p + measurement_noise)
        new_x = x + k * (float(prices[i]) - x)
        dx = 0.5 * dx + 0.5 * (new_x - x)
        x = new_x
        p = (1 - k) * p
        smoothed[i] = x
        slope[i] = dx
    return smoothed, slope


class StrategyV171(TimeSeriesStrategy):
    """
    策略简介：Kalman filter slope + RSI confirmation + OBV trend on 4h.

    使用指标：
    - Kalman Filter: smoothed price and slope for trend direction
    - RSI(14): momentum confirmation
    - OBV: volume flow direction
    - ATR(14): stop distance

    进场条件（做多）：Kalman slope > slope_thresh 且 RSI > 50 且 OBV rising
    进场条件（做空）：Kalman slope < -slope_thresh 且 RSI < 50 且 OBV falling

    出场条件：ATR trailing stop / tiered profit / Kalman slope reversal

    优点：Kalman filter adapts to noise level dynamically
    缺点：Process noise tuning affects responsiveness significantly
    """
    name = "i_alltime_v171"
    warmup = 500
    freq = "4h"

    slope_thresh: float = 0.5
    rsi_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._kalman_slope = None
        self._rsi = None
        self._obv = None
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
        self._atr = atr(highs, lows, closes, period=14)
        _, self._kalman_slope = _kalman_filter(closes)
        self._rsi = rsi(closes, self.rsi_period)
        self._obv = obv(closes, volumes)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        ks = self._kalman_slope[i]
        rsi_val = self._rsi[i]
        if np.isnan(ks) or np.isnan(rsi_val):
            return
        obv_now = self._obv[i]
        obv_prev = self._obv[i - 20] if i >= 20 else np.nan
        if np.isnan(obv_now) or np.isnan(obv_prev):
            return
        obv_up = obv_now > obv_prev

        self.bars_since_last_scale += 1

        # 1. Stop loss
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return
        elif side == -1:
            self.lowest_since_entry = min(self.lowest_since_entry, price)
            trailing = self.lowest_since_entry + self.atr_stop_mult * atr_val
            self.stop_price = min(self.stop_price, trailing)
            if price >= self.stop_price:
                context.close_short()
                self._reset_state()
                return

        # 2. Tiered profit-taking
        if side != 0 and self.entry_price > 0:
            profit_atr = ((price - self.entry_price) / atr_val) if side == 1 else ((self.entry_price - price) / atr_val)
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                cl = max(1, lots // 3)
                (context.close_long if side == 1 else context.close_short)(lots=cl)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                cl = max(1, lots // 3)
                (context.close_long if side == 1 else context.close_short)(lots=cl)
                self._took_profit_3atr = True
                return

        # 3. Signal exit
        if side == 1 and (ks < -self.slope_thresh):
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and (ks > self.slope_thresh):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if ks > self.slope_thresh and rsi_val > 50 and obv_up:
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
            elif ks < -self.slope_thresh and rsi_val < 50 and not obv_up:
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
        elif side != 0 and self._should_add(price, atr_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if side == 1:
                    context.buy(add_lots)
                else:
                    context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1 and price < self.entry_price + atr_val:
            return False
        if self.direction == -1 and price > self.entry_price - atr_val:
            return False
        return True

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_stop_mult * atr_val * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = context.close_raw * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        return max(1, min(risk_lots, int(context.equity * 0.30 / margin)))

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
