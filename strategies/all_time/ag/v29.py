import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.isolation_anomaly import isolation_anomaly
from indicators.regime.market_state import market_state
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class IsolationForestMarketState(TimeSeriesStrategy):
    """
    Isolation Forest Anomaly + Market State strategy for AG futures.

    Fades outlier moves in mean-reverting states (quiet/volatile_range),
    follows outlier moves in trending/breakout states.

    Indicators:
    - Isolation Anomaly (period=120): anomaly score (more negative = more outlier)
    - Market State (period=20): 5-state classifier (quiet/trend_up/trend_down/
      volatile_range/breakout)
    - ATR(14): stop and profit targets

    Entry (Long - fade in ranging):
    - Anomaly score < -0.55 (significant outlier)
    - Market state in {0=quiet, 3=volatile_range}
    - Recent return negative (oversold outlier -> fade)

    Entry (Long - follow in trending):
    - Anomaly score < -0.55
    - Market state in {1=trending_up, 4=breakout}
    - Recent return positive (momentum outlier -> follow)

    Entry (Short): mirror logic for bearish conditions

    Exit:
    - ATR trailing stop
    - Tiered profit-taking at 3ATR, 5ATR
    - Anomaly normalizes (score > -0.45)

    Pros: Adaptive behavior based on market state
    Cons: Market state classification can lag; isolation forest retrain cost
    """
    name = "v29_isolation_forest_market_state"
    warmup = 120 * 3
    freq = "1h"

    iso_period: int = 120
    anomaly_threshold: float = -0.55
    normalize_threshold: float = -0.45
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._anomaly = None
        self._market_state = None
        self._state_conf = None
        self._atr = None
        self._avg_volume = None
        self._ret5 = None

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

        # Build feature matrix
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

        # Use zeros for OI since we don't have it
        oi = np.zeros(n, dtype=np.float64)

        self._anomaly = isolation_anomaly(features, period=self.iso_period)
        self._market_state, self._state_conf = market_state(closes, volumes, oi, period=20)
        self._atr = atr(highs, lows, closes, period=14)

        self._ret5 = np.full(n, np.nan)
        for idx in range(5, n):
            self._ret5[idx] = (closes[idx] - closes[idx - 5]) / closes[idx - 5]

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

        anom = self._anomaly[i]
        ms = self._market_state[i]
        atr_val = self._atr[i]
        ret5 = self._ret5[i]
        if np.isnan(anom) or np.isnan(ms) or np.isnan(atr_val) or np.isnan(ret5):
            return

        is_outlier = anom < self.anomaly_threshold
        is_normal = anom > self.normalize_threshold
        state = int(ms)
        is_ranging = state in (0, 3)   # quiet or volatile_range
        is_trending = state in (1, 2, 4)  # trending or breakout

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

        # Exit when anomaly normalizes
        if (side == 1 or side == -1) and is_normal:
            if side == 1:
                context.close_long()
            else:
                context.close_short()
            self._reset_state()
            return

        # Entry logic
        if side == 0 and is_outlier:
            # Ranging: fade the move
            if is_ranging and ret5 < 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
            elif is_ranging and ret5 > 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
            # Trending: follow the move
            elif is_trending and ret5 > 0 and state in (1, 4):
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
            elif is_trending and ret5 < 0 and state == 2:
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
