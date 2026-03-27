import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.i.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.trend.ichimoku import ichimoku
from indicators.momentum.ppo import ppo
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV145(TimeSeriesStrategy):
    """
    MTF: daily Ichimoku cloud + 4h PPO cross entry.
    Large TF: above cloud. Small TF: PPO cross.
    Strengths: Ichimoku daily filter, PPO normalized. Weaknesses: lagging.
    """
    name = "i_alltime_v145"
    freq = "4h"
    warmup = 500

    ichi_tenkan: int = 9
    ichi_kijun: int = 26
    ppo_fast: int = 12
    ppo_slow: int = 26
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._ppo_line = None
        self._ppo_signal = None
        self._atr = None
        self._avg_volume = None
        self._senkou_a_lg = None
        self._senkou_b_lg = None
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
        pl, ps, _ = ppo(closes, self.ppo_fast, self.ppo_slow, 9)
        self._ppo_line = pl
        self._ppo_signal = ps
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)
        step = 6
        n_lg = n // step
        trim = n_lg * step
        c_lg = closes[:trim].reshape(n_lg, step)[:, -1]
        h_lg = highs[:trim].reshape(n_lg, step).max(axis=1)
        l_lg = lows[:trim].reshape(n_lg, step).min(axis=1)
        _, _, sa, sb, _ = ichimoku(h_lg, l_lg, c_lg, self.ichi_tenkan, self.ichi_kijun, 52)
        self._senkou_a_lg = sa
        self._senkou_b_lg = sb
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
        pl = self._ppo_line[i]
        ps = self._ppo_signal[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(pl) or np.isnan(ps):
            return
        if i < 1:
            return
        pl_p = self._ppo_line[i - 1]
        ps_p = self._ppo_signal[i - 1]
        if np.isnan(pl_p) or np.isnan(ps_p):
            return
        j = self._lg_map[i]
        sa = self._senkou_a_lg[j]
        sb = self._senkou_b_lg[j]
        cl = self._close_lg[j]
        if np.isnan(sa) or np.isnan(sb) or np.isnan(cl):
            return
        cloud_top = max(sa, sb)
        cloud_bot = min(sa, sb)

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
        if side == 1 and not (cl > cloud_top):
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and not (cl < cloud_bot):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if (cl > cloud_top) and (pl_p <= ps_p and pl > ps):
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
            elif (cl < cloud_bot) and (pl_p >= ps_p and pl < ps):
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
