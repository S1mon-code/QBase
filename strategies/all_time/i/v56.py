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
from indicators.momentum.cci import cci
from indicators.volume.klinger import klinger

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV56(TimeSeriesStrategy):
    """
    Bollinger Squeeze release + CCI + Klinger mean-reversion for Iron Ore (4h).

    Indicators:
    - Bollinger Bandwidth: squeeze = low volatility; release after squeeze = mean reversion
    - CCI(20): confirms direction after squeeze release
    - Klinger Oscillator: volume trend confirmation

    Entry (Long): BB bandwidth expanding from squeeze + CCI < -100 + Klinger bullish
    Entry (Short): BB bandwidth expanding from squeeze + CCI > +100 + Klinger bearish
    Exit: CCI returns to zero zone

    Pros: Squeeze release timing catches volatility expansions early
    Cons: Not all squeeze releases lead to mean reversion; some are breakouts
    """
    name = "i_alltime_v56"
    warmup = 400
    freq = "4h"

    bb_period: int = 20
    squeeze_pctl: float = 20.0
    cci_period: int = 20
    cci_threshold: float = 100.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._bb_upper = None
        self._bb_mid = None
        self._bb_lower = None
        self._bandwidth = None
        self._cci = None
        self._kl_line = None
        self._kl_signal = None
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

        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(closes, period=self.bb_period, num_std=2.0)
        n = len(closes)
        self._bandwidth = np.full(n, np.nan)
        valid_mid = self._bb_mid > 1e-9
        self._bandwidth[valid_mid] = (self._bb_upper[valid_mid] - self._bb_lower[valid_mid]) / self._bb_mid[valid_mid]
        self._cci = cci(highs, lows, closes, period=self.cci_period)
        self._kl_line, self._kl_signal = klinger(highs, lows, closes, volumes)
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

        bw = self._bandwidth[i]
        cci_val = self._cci[i]
        kl = self._kl_line[i]
        ks = self._kl_signal[i]
        c = context.get_full_close_array()[i]
        bb_m = self._bb_mid[i]
        atr_val = self._atr[i]
        if np.isnan(bw) or np.isnan(cci_val) or np.isnan(kl) or np.isnan(ks) or np.isnan(atr_val):
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
        if side == 1 and cci_val > -50:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and cci_val < 50:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        # Detect squeeze release: bandwidth was in low percentile recently, now expanding
        lookback = 120
        start_idx = max(0, i - lookback)
        bw_hist = self._bandwidth[start_idx:i + 1]
        valid_bw = bw_hist[~np.isnan(bw_hist)]
        squeeze_release = False
        if len(valid_bw) >= 20:
            pctl = np.percentile(valid_bw, self.squeeze_pctl)
            # Was in squeeze 5 bars ago, now above squeeze
            bw_5ago = self._bandwidth[max(0, i - 5)]
            if not np.isnan(bw_5ago) and bw_5ago < pctl and bw > pctl:
                squeeze_release = True
        if side == 0 and squeeze_release and cci_val < -self.cci_threshold and kl > ks:
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
        elif side == 0 and squeeze_release and cci_val > self.cci_threshold and kl < ks:
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
        elif side == 1 and self._should_add_long(price, atr_val, cci_val < -50):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self._should_add_short(price, atr_val, cci_val > 50):
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
