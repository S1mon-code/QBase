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
from indicators.trend.supertrend import supertrend
from indicators.trend.vortex import vortex
from indicators.trend.ema import ema
from indicators.volume.force_index import force_index

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV184(TimeSeriesStrategy):
    """
    策略简介：Vote(Supertrend, Vortex VI+>VI-, EMA20>EMA50) >= 2/3 + Force Index.

    使用指标：
    - Supertrend(10,3): trend direction
    - Vortex(14): VI+ vs VI- comparison
    - EMA(20) vs EMA(50): trend cross
    - Force Index(13): volume-price force confirmation
    - ATR(14): stop distance

    进场条件（做多）：>= 2/3 bullish votes 且 Force Index > 0
    进场条件（做空）：>= 2/3 bearish votes 且 Force Index < 0

    出场条件：ATR trailing stop / tiered profit / vote reversal

    优点：Three independent trend methods provide robust direction signal
    缺点：All trend-following, no mean-reversion component
    """
    name = "i_alltime_v184"
    warmup = 400
    freq = "4h"

    ema_fast: int = 20
    ema_slow: int = 50
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._st_dir = None
        self._vi_plus = None
        self._vi_minus = None
        self._ema_fast = None
        self._ema_slow = None
        self._fi = None
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
        self._avg_volume = fast_avg_volume(volumes, 20)
        _, self._st_dir = supertrend(highs, lows, closes, 10, 3.0)
        self._vi_plus, self._vi_minus = vortex(highs, lows, closes, 14)
        self._ema_fast = ema(closes, self.ema_fast)
        self._ema_slow = ema(closes, self.ema_slow)
        self._fi = force_index(closes, volumes, 13)

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

        st = self._st_dir[i]
        vp = self._vi_plus[i]
        vm = self._vi_minus[i]
        ef = self._ema_fast[i]
        es = self._ema_slow[i]
        fi_val = self._fi[i]
        if np.isnan(st) or np.isnan(vp) or np.isnan(vm) or np.isnan(ef) or np.isnan(es) or np.isnan(fi_val):
            return

        bull_votes = 0
        bear_votes = 0
        if st == 1: bull_votes += 1
        elif st == -1: bear_votes += 1
        if vp > vm: bull_votes += 1
        elif vm > vp: bear_votes += 1
        if ef > es: bull_votes += 1
        elif ef < es: bear_votes += 1
        vol_ok = (fi_val > 0) if bull_votes >= 2 else ((fi_val < 0) if bear_votes >= 2 else False)

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
        if side == 1 and bear_votes >= 2:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and bull_votes >= 2:
            context.close_short()
            self._reset_state()
            return


        # 4. Entry
        if side == 0:
            if bull_votes >= 2 and vol_ok:
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
            elif bear_votes >= 2 and vol_ok:
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
