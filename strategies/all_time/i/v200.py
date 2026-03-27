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
from indicators.volatility.vol_ratio import volatility_ratio
from indicators.trend.keltner import keltner
from indicators.momentum.fisher_transform import fisher_transform
from indicators.volume.oi_momentum import oi_momentum

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV200(TimeSeriesStrategy):
    """
    策略简介：Dual-mode: Vol Ratio high→breakout(Keltner channel), low→revert(Fisher Transform) + OI Momentum on 1h.

    使用指标：
    - Volatility Ratio(14): high(>1.5)=breakout mode, low(<0.8)=range mode
    - Keltner(20,10,1.5): channel breakout in breakout mode
    - Fisher Transform(10): overbought/oversold oscillator in range mode
    - OI Momentum(20): open interest trend confirmation
    - ATR(14): stop distance

    进场条件（突破-做多）：VR>1.5 且 price>Keltner upper 且 OI mom>0
    进场条件（突破-做空）：VR>1.5 且 price<Keltner lower 且 OI mom>0
    进场条件（回归-做多）：VR<0.8 且 Fisher<-1.5 且 OI mom>0
    进场条件（回归-做空）：VR<0.8 且 Fisher>1.5 且 OI mom>0

    出场条件：ATR trailing stop / tiered profit / mode-specific exit

    优点：Keltner channel adapts to volatility, Fisher normalizes extremes
    缺点：Fisher Transform can give premature signals in strong trends
    """
    name = "i_alltime_v200"
    warmup = 600
    freq = "1h"

    vr_breakout: float = 1.5
    vr_range: float = 0.8
    fisher_thresh: float = 1.5
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._vr = None
        self._kc_upper = None
        self._kc_lower = None
        self._kc_mid = None
        self._fisher = None
        self._oi_mom = None
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
        self._atr = atr(highs, lows, closes, period=14)
        self._vr = volatility_ratio(highs, lows, closes, 14)
        self._kc_upper, self._kc_mid, self._kc_lower = keltner(highs, lows, closes, 20, 10, 1.5)
        self._fisher, _ = fisher_transform(highs, lows, 10)
        self._oi_mom = oi_momentum(oi, 20)
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
        vr = self._vr[i]
        kcu = self._kc_upper[i]
        kcl = self._kc_lower[i]
        kcm = self._kc_mid[i]
        fv = self._fisher[i]
        oi_val = self._oi_mom[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(vr) or np.isnan(kcu) or np.isnan(kcl) or np.isnan(fv) or np.isnan(oi_val):
            return

        is_breakout = vr > self.vr_breakout
        is_range = vr < self.vr_range

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
            if is_breakout and price < kcl:
                context.close_long()
                self._reset_state()
                return
            elif is_range and fv > 0:
                context.close_long()
                self._reset_state()
                return
        elif side == -1:
            if is_breakout and price > kcu:
                context.close_short()
                self._reset_state()
                return
            elif is_range and fv < 0:
                context.close_short()
                self._reset_state()
                return

        # 4. Entry
        if side == 0:
            if (is_breakout and price > kcu and oi_val > 0) or (is_range and fv < -self.fisher_thresh and oi_val > 0):
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
            elif (is_breakout and price < kcl and oi_val > 0) or (is_range and fv > self.fisher_thresh and oi_val > 0):
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
