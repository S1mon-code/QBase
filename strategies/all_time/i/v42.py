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
from indicators.trend.keltner import keltner
from indicators.momentum.stochastic import stochastic
from indicators.volume.mfi import mfi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV42(TimeSeriesStrategy):
    """
    Keltner Channel + Stochastic(14,3) + MFI mean-reversion for Iron Ore.

    Indicators:
    - Keltner Channel(20, 1.5): ATR-based bands for overbought/oversold zones
    - Stochastic(14,3): momentum oscillator for reversal timing
    - MFI(14): money flow index confirms volume-weighted buying/selling pressure

    Entry (Long):
    - Price below lower Keltner Channel
    - Stochastic %K < 20 (oversold)
    - MFI < 30 (money outflow exhaustion)

    Entry (Short):
    - Price above upper Keltner Channel
    - Stochastic %K > 80 (overbought)
    - MFI > 70 (money inflow exhaustion)

    Exit:
    - Price returns to Keltner middle line
    - ATR trailing stop / tiered profit-taking

    Pros: Keltner adapts to volatility better than fixed BB; MFI adds volume weight
    Cons: Keltner channels can widen in volatile markets reducing signal frequency
    """
    name = "i_alltime_v42"
    warmup = 250
    freq = "daily"

    kc_period: int = 20
    kc_mult: float = 1.5
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._kc_upper = None
        self._kc_mid = None
        self._kc_lower = None
        self._stoch_k = None
        self._stoch_d = None
        self._mfi = None
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

        self._kc_upper, self._kc_mid, self._kc_lower = keltner(
            highs, lows, closes, period=self.kc_period, mult=self.kc_mult
        )
        self._stoch_k, self._stoch_d = stochastic(highs, lows, closes, k_period=14, d_period=3)
        self._mfi = mfi(highs, lows, closes, volumes, period=14)
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

        kc_upper = self._kc_upper[i]
        kc_mid = self._kc_mid[i]
        kc_lower = self._kc_lower[i]
        stoch_k = self._stoch_k[i]
        mfi_val = self._mfi[i]
        atr_val = self._atr[i]
        if np.isnan(kc_upper) or np.isnan(stoch_k) or np.isnan(mfi_val) or np.isnan(atr_val):
            return

        closes = context.get_full_close_array()
        c = closes[i]
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

        # 3. Signal exit — price returns to Keltner middle
        if side == 1 and c >= kc_mid:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and c <= kc_mid:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry - Long
        if side == 0 and c < kc_lower and stoch_k < self.stoch_oversold and mfi_val < 30:
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

        # 4. Entry - Short
        elif side == 0 and c > kc_upper and stoch_k > self.stoch_overbought and mfi_val > 70:
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
        elif side == 1 and self._should_add_long(price, atr_val, stoch_k < 30):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self._should_add_short(price, atr_val, stoch_k > 70):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add_long(self, price, atr_val, signal_confirm):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        return signal_confirm

    def _should_add_short(self, price, atr_val, signal_confirm):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
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
