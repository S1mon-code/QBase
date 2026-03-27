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
from indicators.momentum.rsi import rsi
from indicators.volume.obv import obv

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV41(TimeSeriesStrategy):
    """
    Bollinger Bands + RSI(14) + OBV mean-reversion strategy for Iron Ore.

    Indicators:
    - Bollinger Bands(20, 2.0): upper/lower bands define overbought/oversold zones
    - RSI(14): confirms oversold (<30) or overbought (>70) conditions
    - OBV: volume confirmation — rising OBV on oversold = accumulation

    Entry (Long):
    - Price closes below lower Bollinger Band
    - RSI < rsi_oversold threshold
    - OBV rising (current > OBV 10 bars ago)

    Entry (Short):
    - Price closes above upper Bollinger Band
    - RSI > rsi_overbought threshold
    - OBV falling (current < OBV 10 bars ago)

    Exit:
    - Price returns to middle Bollinger Band (mean)
    - ATR trailing stop
    - Tiered profit-taking at 3ATR / 5ATR

    Pros: Classic mean-reversion with triple confirmation reduces false signals
    Cons: Struggles in strong trending markets where price rides the band
    """
    name = "i_alltime_v41"
    warmup = 250
    freq = "daily"

    bb_period: int = 20
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    atr_stop_mult: float = 3.0
    obv_lookback: int = 10

    def __init__(self):
        super().__init__()
        self._bb_upper = None
        self._bb_mid = None
        self._bb_lower = None
        self._rsi = None
        self._obv = None
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

        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            closes, period=self.bb_period, num_std=2.0
        )
        self._rsi = rsi(closes, period=14)
        self._obv = obv(closes, volumes)
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

        bb_upper = self._bb_upper[i]
        bb_mid = self._bb_mid[i]
        bb_lower = self._bb_lower[i]
        rsi_val = self._rsi[i]
        atr_val = self._atr[i]
        obv_val = self._obv[i]
        if np.isnan(bb_upper) or np.isnan(rsi_val) or np.isnan(atr_val) or np.isnan(obv_val):
            return

        obv_prev = self._obv[max(0, i - self.obv_lookback)]
        if np.isnan(obv_prev):
            return
        obv_rising = obv_val > obv_prev
        obv_falling = obv_val < obv_prev

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

        # 3. Signal exit — price returns to mean
        if side == 1 and c >= bb_mid:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and c <= bb_mid:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry - Long
        if side == 0 and c < bb_lower and rsi_val < self.rsi_oversold and obv_rising:
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
        elif side == 0 and c > bb_upper and rsi_val > self.rsi_overbought and obv_falling:
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

        # 5. Scale-in - Long
        elif side == 1 and self._should_add_long(price, atr_val, c < bb_lower):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

        # 5. Scale-in - Short
        elif side == -1 and self._should_add_short(price, atr_val, c > bb_upper):
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
        if not signal_confirm:
            return False
        return True

    def _should_add_short(self, price, atr_val, signal_confirm):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price > self.entry_price - atr_val:
            return False
        if not signal_confirm:
            return False
        return True

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
