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
from indicators.trend.adx import adx
from indicators.trend.aroon import aroon
from indicators.momentum.stochastic import stochastic
from indicators.volume.cmf import cmf

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV193(TimeSeriesStrategy):
    """
    策略简介：Dual-mode: ADX>25→trend(Aroon direction), ADX<20→revert(Stochastic) + CMF.

    使用指标：
    - ADX(14): regime switch
    - Aroon(25): trend direction in trend mode
    - Stochastic(14,3): overbought/oversold in range mode
    - CMF(20): money flow confirmation
    - ATR(14): stop distance

    进场条件（趋势-做多）：ADX>25 且 Aroon osc>50 且 CMF>0
    进场条件（趋势-做空）：ADX>25 且 Aroon osc<-50 且 CMF<0
    进场条件（回归-做多）：ADX<20 且 Stoch<20 且 CMF>0
    进场条件（回归-做空）：ADX<20 且 Stoch>80 且 CMF<0

    出场条件：ATR trailing stop / tiered profit / mode exit

    优点：Aroon gives clear directional bias in trending markets
    缺点：Dead zone between ADX 20-25 misses signals
    """
    name = "i_alltime_v193"
    warmup = 400
    freq = "4h"

    adx_trend: float = 25.0
    adx_range: float = 20.0
    stoch_os: float = 20.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._adx_val = None
        self._aroon_osc = None
        self._stoch_k = None
        self._cmf = None
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
        self._adx_val = adx(highs, lows, closes, 14)
        _, _, self._aroon_osc = aroon(highs, lows, 25)
        self._stoch_k, _ = stochastic(highs, lows, closes, 14, 3)
        self._cmf = cmf(highs, lows, closes, volumes, 20)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        adx_v = self._adx_val[i]
        ao = self._aroon_osc[i]
        sk = self._stoch_k[i]
        cmf_val = self._cmf[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(adx_v) or np.isnan(ao) or np.isnan(sk) or np.isnan(cmf_val):
            return

        is_trend = adx_v > self.adx_trend
        is_range = adx_v < self.adx_range

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
        if side == 1:
            if is_trend and ao < -50:
                context.close_long()
                self._reset_state()
                return
            elif is_range and sk > 80:
                context.close_long()
                self._reset_state()
                return
        elif side == -1:
            if is_trend and ao > 50:
                context.close_short()
                self._reset_state()
                return
            elif is_range and sk < 20:
                context.close_short()
                self._reset_state()
                return

        # 4. Entry
        if side == 0:
            if (is_trend and ao > 50 and cmf_val > 0) or (is_range and sk < self.stoch_os and cmf_val > 0):
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
            elif (is_trend and ao < -50 and cmf_val < 0) or (is_range and sk > (100 - self.stoch_os) and cmf_val < 0):
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
        if side != 0 and self._should_add(price, atr_val):
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
