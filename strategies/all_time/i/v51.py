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
from indicators.momentum.stoch_rsi import stoch_rsi
from indicators.volatility.bollinger import bollinger_bands
from indicators.volume.oi_momentum import oi_momentum

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV51(TimeSeriesStrategy):
    """
    Stochastic RSI + Bollinger %B + OI Momentum mean-reversion for Iron Ore (4h).

    Indicators:
    - Stochastic RSI(14,14,3,3): double-smoothed RSI oscillator for precise timing
    - Bollinger %B(20,2): normalized position in bands
    - OI Momentum(10): open interest rate of change; rising OI confirms conviction

    Entry (Long): Stoch RSI %K < 0.2 + %B < 0.1 + OI momentum positive
    Entry (Short): Stoch RSI %K > 0.8 + %B > 0.9 + OI momentum positive
    Exit: Stoch RSI crosses 0.5

    Pros: Stoch RSI is more sensitive than regular RSI for timing
    Cons: Overly sensitive oscillator may trigger premature entries
    """
    name = "i_alltime_v51"
    warmup = 400
    freq = "4h"

    srsi_oversold: float = 0.2
    srsi_overbought: float = 0.8
    bb_period: int = 20
    oi_mom_period: int = 10
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._srsi_k = None
        self._srsi_d = None
        self._bb_upper = None
        self._bb_mid = None
        self._bb_lower = None
        self._bb_pctb = None
        self._oi_mom = None
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
        self.direction = 0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()
        self._srsi_k, self._srsi_d = stoch_rsi(closes, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3)
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(closes, period=self.bb_period, num_std=2.0)
        n = len(closes)
        self._bb_pctb = np.full(n, np.nan)
        bw = self._bb_upper - self._bb_lower
        valid = bw > 1e-9
        self._bb_pctb[valid] = (closes[valid] - self._bb_lower[valid]) / bw[valid]
        self._oi_mom = oi_momentum(oi, period=self.oi_mom_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        vol = context.volume
        if self._avg_volume[i] is not None and not np.isnan(self._avg_volume[i]):
            if vol < self._avg_volume[i] * 0.1:
                return

        srsi_k = self._srsi_k[i]
        pctb = self._bb_pctb[i]
        oi_m = self._oi_mom[i]
        atr_val = self._atr[i]
        if np.isnan(srsi_k) or np.isnan(pctb) or np.isnan(oi_m) or np.isnan(atr_val):
            return

        self.bars_since_last_scale += 1

        # 1. Stop loss - Long
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return
        # 1. Stop loss - Short
        if side == -1:
            self.lowest_since_entry = min(self.lowest_since_entry, price)
            trailing = self.lowest_since_entry + self.atr_stop_mult * atr_val
            self.stop_price = min(self.stop_price, trailing)
            if price >= self.stop_price:
                context.close_short()
                self._reset_state()
                return

        # 2. Tiered profit - Long
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
        # 2. Tiered profit - Short
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

        # 3. Signal exit
        if side == 1 and srsi_k >= 0.5:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and srsi_k <= 0.5:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and srsi_k < self.srsi_oversold and pctb < 0.1 and oi_m > 0:
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
        elif side == 0 and srsi_k > self.srsi_overbought and pctb > 0.9 and oi_m > 0:
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
        elif side == 1 and self._should_add_long(price, atr_val, srsi_k < 0.3):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self._should_add_short(price, atr_val, srsi_k > 0.7):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add_long(self, price, atr_val, signal_confirm):
        if self.position_scale >= MAX_SCALE or self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        return signal_confirm

    def _should_add_short(self, price, atr_val, signal_confirm):
        if self.position_scale >= MAX_SCALE or self.bars_since_last_scale < 10:
            return False
        if price > self.entry_price - atr_val:
            return False
        return signal_confirm

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
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
        self.direction = 0
