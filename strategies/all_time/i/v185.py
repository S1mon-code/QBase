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
from indicators.trend.ichimoku import ichimoku
from indicators.momentum.macd import macd
from indicators.trend.adx import adx
from indicators.volume.volume_spike import volume_spike

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV185(TimeSeriesStrategy):
    """
    策略简介：Vote(Ichimoku cloud position, MACD hist, ADX trending) >= 2/3 + Volume Spike.

    使用指标：
    - Ichimoku(9,26,52): price above/below cloud
    - MACD(12,26,9): histogram direction
    - ADX(14): trend strength > 25
    - Volume Spike(20,1.5): breakout volume confirmation
    - ATR(14): stop distance

    进场条件（做多）：>= 2/3 bullish votes 且 Volume Spike detected
    进场条件（做空）：>= 2/3 bearish votes 且 Volume Spike detected

    出场条件：ATR trailing stop / tiered profit / vote reversal

    优点：Ichimoku provides multi-dimensional trend assessment
    缺点：Ichimoku needs large warmup (52 periods), slower reaction
    """
    name = "i_alltime_v185"
    warmup = 500
    freq = "4h"

    adx_threshold: float = 25.0
    vol_spike_mult: float = 1.5
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._senkou_a = None
        self._senkou_b = None
        self._macd_hist = None
        self._adx_val = None
        self._vol_spike = None
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
        _, _, self._senkou_a, self._senkou_b, _ = ichimoku(highs, lows, closes)
        _, _, self._macd_hist = macd(closes, 12, 26, 9)
        self._adx_val = adx(highs, lows, closes, 14)
        self._vol_spike = volume_spike(volumes, 20, self.vol_spike_mult)

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

        sa = self._senkou_a[i]
        sb = self._senkou_b[i]
        mh = self._macd_hist[i]
        adx_v = self._adx_val[i]
        if np.isnan(sa) or np.isnan(sb) or np.isnan(mh) or np.isnan(adx_v):
            return

        cloud_top = max(sa, sb)
        cloud_bot = min(sa, sb)
        bull_votes = 0
        bear_votes = 0
        if price > cloud_top: bull_votes += 1
        elif price < cloud_bot: bear_votes += 1
        if mh > 0: bull_votes += 1
        elif mh < 0: bear_votes += 1
        if adx_v > self.adx_threshold:
            if mh > 0: bull_votes += 1
            elif mh < 0: bear_votes += 1
        vs = self._vol_spike[i]
        vol_ok = bool(vs)

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
