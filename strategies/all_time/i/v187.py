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
from indicators.trend.adx import adx_with_di
from indicators.momentum.stochastic import stochastic
from indicators.momentum.ppo import ppo
from indicators.volume.obv import obv

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV187(TimeSeriesStrategy):
    """
    策略简介：Vote(ADX DI direction, Stochastic momentum, PPO hist) >= 2/3 + OBV.

    使用指标：
    - ADX with DI(14): +DI > -DI = bullish
    - Stochastic(14,3): > 50 = bullish momentum
    - PPO(12,26,9): histogram direction
    - OBV: volume flow confirmation
    - ATR(14): stop distance

    进场条件（做多）：>= 2/3 bullish votes 且 OBV rising
    进场条件（做空）：>= 2/3 bearish votes 且 OBV falling

    出场条件：ATR trailing stop / tiered profit / vote reversal

    优点：DI direction adds directional strength to ADX trend strength
    缺点：Multiple lagging indicators may delay entries
    """
    name = "i_alltime_v187"
    warmup = 600
    freq = "1h"

    stoch_period: int = 14
    ppo_fast: int = 12
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._adx_val = None
        self._plus_di = None
        self._minus_di = None
        self._stoch_k = None
        self._ppo_hist = None
        self._obv = None
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
        self._adx_val, self._plus_di, self._minus_di = adx_with_di(highs, lows, closes, 14)
        self._stoch_k, _ = stochastic(highs, lows, closes, self.stoch_period, 3)
        _, _, self._ppo_hist = ppo(closes, self.ppo_fast, 26, 9)
        self._obv = obv(closes, volumes)

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

        pdi = self._plus_di[i]
        mdi = self._minus_di[i]
        sk = self._stoch_k[i]
        ph = self._ppo_hist[i]
        if np.isnan(pdi) or np.isnan(mdi) or np.isnan(sk) or np.isnan(ph):
            return

        bull_votes = 0
        bear_votes = 0
        if pdi > mdi: bull_votes += 1
        elif mdi > pdi: bear_votes += 1
        if sk > 50: bull_votes += 1
        elif sk < 50: bear_votes += 1
        if ph > 0: bull_votes += 1
        elif ph < 0: bear_votes += 1
        obv_now = self._obv[i]
        obv_prev = self._obv[i - 20] if i >= 20 else np.nan
        if np.isnan(obv_now) or np.isnan(obv_prev):
            return
        vol_ok = (obv_now > obv_prev) if bull_votes >= 2 else ((obv_now < obv_prev) if bear_votes >= 2 else False)

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
