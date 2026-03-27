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
from indicators.volatility.bollinger import bollinger_bands
from indicators.momentum.williams_r import williams_r
from indicators.volume.cmf import cmf

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV44(TimeSeriesStrategy):
    """
    Williams %R + Bollinger %B + CMF mean-reversion for Iron Ore.

    Indicators:
    - Williams %R(14): momentum oscillator; <-80 oversold, >-20 overbought
    - Bollinger %B(20,2): normalized position within bands; <0 below lower, >1 above upper
    - CMF(20): Chaikin Money Flow; positive=accumulation, negative=distribution

    Entry (Long):
    - Williams %R < wr_oversold (deeply oversold)
    - Bollinger %B < 0.0 (below lower band)
    - CMF > 0 (smart money accumulating despite price drop)

    Entry (Short):
    - Williams %R > wr_overbought (deeply overbought)
    - Bollinger %B > 1.0 (above upper band)
    - CMF < 0 (distribution despite price rise)

    Exit:
    - Williams %R crosses -50 (midpoint) OR Bollinger %B returns to 0.5
    - ATR trailing stop / tiered profit-taking

    Pros: CMF divergence from price is a powerful reversal signal
    Cons: CMF can give false divergences during accumulation/distribution phases
    """
    name = "i_alltime_v44"
    warmup = 250
    freq = "daily"

    wr_oversold: float = -80.0
    wr_overbought: float = -20.0
    bb_period: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._wr = None
        self._bb_upper = None
        self._bb_mid = None
        self._bb_lower = None
        self._cmf = None
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

        self._wr = williams_r(highs, lows, closes, period=14)
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            closes, period=self.bb_period, num_std=2.0
        )
        self._cmf = cmf(highs, lows, closes, volumes, period=20)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        # Compute Bollinger %B
        n = len(closes)
        self._bb_pctb = np.full(n, np.nan)
        band_width = self._bb_upper - self._bb_lower
        valid = band_width > 1e-9
        self._bb_pctb[valid] = (closes[valid] - self._bb_lower[valid]) / band_width[valid]

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

        wr_val = self._wr[i]
        pctb = self._bb_pctb[i]
        cmf_val = self._cmf[i]
        atr_val = self._atr[i]
        if np.isnan(wr_val) or np.isnan(pctb) or np.isnan(cmf_val) or np.isnan(atr_val):
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

        if side == -1:
            self.lowest_since_entry = min(self.lowest_since_entry, price)
            trailing = self.lowest_since_entry + self.atr_stop_mult * atr_val
            self.stop_price = min(self.stop_price, trailing)
            if price >= self.stop_price:
                context.close_short()
                self._reset_state()
                return

        # 2. Tiered profit
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

        # 3. Signal exit — Williams %R crosses midpoint or %B returns to 0.5
        if side == 1 and (wr_val > -50 or pctb > 0.5):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (wr_val < -50 or pctb < 0.5):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry - Long
        if side == 0 and wr_val < self.wr_oversold and pctb < 0.0 and cmf_val > 0:
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

        elif side == 0 and wr_val > self.wr_overbought and pctb > 1.0 and cmf_val < 0:
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
        elif side == 1 and self._should_add_long(price, atr_val, wr_val < -60):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self._should_add_short(price, atr_val, wr_val > -40):
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
