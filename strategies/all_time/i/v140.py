import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.i.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.trend.donchian import donchian
from indicators.momentum.stochastic import stochastic
from indicators.volume.oi_momentum import oi_momentum
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV140(TimeSeriesStrategy):
    """
    MTF: 4h Donchian + 1h Stochastic + OI Momentum entry.
    Large TF: close > Don mid. Small TF: Stoch oversold + OI mom > 0.
    Strengths: OI unique futures signal. Weaknesses: OI sparse, Stoch whipsaw.
    """
    name = "i_alltime_v140"
    freq = "1h"
    warmup = 800

    don_period: int = 20
    stoch_period: int = 14
    stoch_entry: float = 20.0
    oi_mom_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._stoch_k = None
        self._oi_mom = None
        self._atr = None
        self._avg_volume = None
        self._don_mid_lg = None
        self._close_lg = None
        self._lg_map = None

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
        n = len(closes)
        self._stoch_k, _ = stochastic(highs, lows, closes, self.stoch_period)
        try:
            oi_arr = context.get_full_oi_array()
            self._oi_mom = oi_momentum(oi_arr, self.oi_mom_period)
        except Exception:
            self._oi_mom = np.full(n, 0.0)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)
        step = 4
        n_lg = n // step
        trim = n_lg * step
        c_lg = closes[:trim].reshape(n_lg, step)[:, -1]
        h_lg = highs[:trim].reshape(n_lg, step).max(axis=1)
        l_lg = lows[:trim].reshape(n_lg, step).min(axis=1)
        _, _, mid = donchian(h_lg, l_lg, self.don_period)
        self._don_mid_lg = mid
        self._close_lg = c_lg
        self._lg_map = np.maximum(0, (np.arange(n) + 1) // step - 1)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        sk = self._stoch_k[i]
        oi_m = self._oi_mom[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(sk) or np.isnan(oi_m):
            return
        j = self._lg_map[i]
        d_mid = self._don_mid_lg[j]
        cl = self._close_lg[j]
        if np.isnan(d_mid) or np.isnan(cl):
            return

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

        # 3. Signal exit: large TF reversal
        if side == 1 and not (cl > d_mid):
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and not (cl < d_mid):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if (cl > d_mid) and (sk < self.stoch_entry and oi_m > 0):
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
            elif (cl < d_mid) and (sk > (100 - self.stoch_entry) and oi_m < 0):
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
                (context.buy if side == 1 else context.sell)(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val):
        if self.position_scale >= MAX_SCALE or self.bars_since_last_scale < 10:
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
