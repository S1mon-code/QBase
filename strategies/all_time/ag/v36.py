import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.copula_tail import tail_dependence
from indicators.regime.vol_regime_markov import vol_regime_simple
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class CopulaDependencyMacroRegime(TimeSeriesStrategy):
    """
    Copula Dependency (Tail Dependence) + Macro Regime strategy for AG futures.

    Trades copula tail dependence shifts between price returns and volume
    returns, aligned with risk-on (low vol) / risk-off (high vol) regime.

    Indicators:
    - Tail Dependence (period=120): lower/upper tail dependence between
      price returns and volume returns
    - Vol Regime Simple (period=60): 0=low vol (risk-on), 1=high vol (risk-off)
    - ATR(14): stop and profit targets

    Entry (Long - risk-on):
    - Vol regime = 0 (low vol / risk-on)
    - Upper tail dependence rising (price and volume rally together)
    - Lower tail dependence low (no crash synchronization)

    Entry (Short - risk-off):
    - Vol regime = 1 (high vol / risk-off)
    - Lower tail dependence rising (crash correlation increasing)
    - Upper tail dependence low

    Exit:
    - ATR trailing stop
    - Tiered profit-taking at 3ATR, 5ATR
    - Regime flip

    Pros: Tail dependence captures nonlinear co-movement; macro regime adds context
    Cons: Tail dependence estimate noisy with small samples; vol regime lags
    """
    name = "v36_copula_macro_regime"
    warmup = 120 * 3
    freq = "daily"

    tail_period: int = 120
    vol_period: int = 60
    tail_threshold: float = 0.3
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._lower_tail = None
        self._upper_tail = None
        self._vol_regime = None
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

        n = len(closes)
        safe_c = np.maximum(closes, 1e-9)
        safe_v = np.maximum(volumes, 1e-9)

        price_ret = np.full(n, np.nan)
        price_ret[1:] = np.log(safe_c[1:] / safe_c[:-1])

        vol_ret = np.full(n, np.nan)
        vol_ret[1:] = np.log(safe_v[1:] / safe_v[:-1])

        self._lower_tail, self._upper_tail = tail_dependence(
            price_ret, vol_ret, period=self.tail_period
        )
        self._vol_regime, _, _ = vol_regime_simple(closes, period=self.vol_period)
        self._atr = atr(highs, lows, closes, period=14)

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

        lt = self._lower_tail[i]
        ut = self._upper_tail[i]
        vr = self._vol_regime[i]
        atr_val = self._atr[i]
        if np.isnan(lt) or np.isnan(ut) or np.isnan(atr_val):
            return

        risk_on = not np.isnan(vr) and vr == 0.0
        risk_off = not np.isnan(vr) and vr == 1.0

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

        # Exit on regime flip
        if side == 1 and risk_off:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and risk_on:
            context.close_short()
            self._reset_state()
            return

        # Entry - Long (risk-on with upper tail dependence)
        if side == 0 and risk_on and ut > self.tail_threshold and lt < self.tail_threshold:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.lowest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # Entry - Short (risk-off with lower tail dependence)
        elif side == 0 and risk_off and lt > self.tail_threshold and ut < self.tail_threshold:
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
        elif side == 1 and self._should_add_long(price, atr_val, risk_on):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

        # Scaling - Short
        elif side == -1 and self._should_add_short(price, atr_val, risk_off):
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
