import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.regime.changepoint import changepoint_score
from indicators.regime.vol_regime_markov import vol_regime_simple
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BayesianChangepointVolState(TimeSeriesStrategy):
    """
    Bayesian Changepoint + Volatility State Transition for AG futures.

    Enters when a changepoint aligns with a volatility state transition,
    catching regime shifts early.

    Indicators:
    - Changepoint Score (period=60): probability of structural break (0-1)
    - Vol Regime Simple (period=60): 2-state high/low vol classification
    - ATR(14): stop and profit targets

    Entry (Long):
    - Changepoint score > 0.6 (likely structural break)
    - Vol regime just transitioned (regime != previous bar regime)
    - Recent momentum positive (5-bar return > 0)

    Entry (Short):
    - Changepoint score > 0.6
    - Vol regime transition detected
    - Recent momentum negative

    Exit:
    - ATR trailing stop
    - Tiered profit-taking at 3ATR, 5ATR
    - Changepoint score drops and momentum reverses

    Pros: Captures regime shifts early; vol state adds confirmation
    Cons: Changepoint detection can be noisy; may trigger on false breaks
    """
    name = "v26_changepoint_vol_state"
    warmup = 60 * 3
    freq = "4h"

    cp_period: int = 60
    vol_period: int = 60
    cp_threshold: float = 0.6
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._cp_score = None
        self._vol_regime = None
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
        self._cp_score = changepoint_score(closes, period=self.cp_period)
        self._vol_regime, _, _ = vol_regime_simple(closes, period=self.vol_period)
        self._atr = atr(highs, lows, closes, period=14)

        # 5-bar momentum
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

        cp_val = self._cp_score[i]
        vol_reg = self._vol_regime[i]
        atr_val = self._atr[i]
        mom_val = self._mom5[i]

        if np.isnan(cp_val) or np.isnan(atr_val) or np.isnan(mom_val):
            return

        # Detect vol regime transition
        vol_transition = False
        if i > 0 and not np.isnan(vol_reg) and not np.isnan(self._vol_regime[i - 1]):
            vol_transition = vol_reg != self._vol_regime[i - 1]

        has_changepoint = cp_val > self.cp_threshold
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

        # Signal exit: momentum reverses
        if side == 1 and mom_val < 0:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and mom_val > 0:
            context.close_short()
            self._reset_state()
            return

        # Entry - Long
        if side == 0 and has_changepoint and vol_transition and mom_val > 0:
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
        elif side == 0 and has_changepoint and vol_transition and mom_val < 0:
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
        elif side == 1 and self._should_add_long(price, atr_val, mom_val > 0):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

        # Scaling - Short
        elif side == -1 and self._should_add_short(price, atr_val, mom_val < 0):
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
