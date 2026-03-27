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
from indicators.momentum.macd import macd
from indicators.volume.volume_spike import volume_spike

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV27(TimeSeriesStrategy):
    """
    策略简介：ADX+DI方向 + MACD柱确认 + 放量触发的1h趋势策略。

    使用指标：
    - ADX(14) + DI: 趋势强度和方向
    - MACD(12,26,9): 动量方向确认
    - Volume Spike(20,1.5): 放量确认

    进场条件（做多）：ADX>20 且 +DI>-DI 且 MACD柱>0 且 放量
    进场条件（做空）：ADX>20 且 -DI>+DI 且 MACD柱<0 且 放量

    出场条件：ATR追踪止损 / 分层止盈 / DI交叉或MACD反转

    优点：ADX+DI组合精确判断趋势方向和强度
    缺点：1h级别ADX波动大，放量条件可能过于严格
    """
    name = "i_alltime_v27"
    warmup = 800
    freq = "1h"

    adx_period: int = 14
    adx_threshold: float = 20.0
    macd_fast: int = 12
    macd_slow: int = 26
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
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
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        self._adx, self._plus_di, self._minus_di = adx(highs, lows, closes, period=self.adx_period)
        _, _, self._macd_hist = macd(closes, self.macd_fast, self.macd_slow, 9)
        self._vol_spike = volume_spike(volumes, 20, 1.5)

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

        adx_val = self._adx[i]
        pdi = self._plus_di[i]
        mdi = self._minus_di[i]
        hist = self._macd_hist[i]
        spike = self._vol_spike[i]
        if np.isnan(adx_val) or np.isnan(pdi) or np.isnan(mdi) or np.isnan(hist) or np.isnan(spike):
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
                if side == 1: context.close_long(lots=cl)
                else: context.close_short(lots=cl)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                cl = max(1, lots // 3)
                if side == 1: context.close_long(lots=cl)
                else: context.close_short(lots=cl)
                self._took_profit_3atr = True
                return

        # 3. Signal-based exit
        if side == 1 and (mdi > pdi or hist < 0):
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and (pdi > mdi or hist > 0):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if adx_val > self.adx_threshold and pdi > mdi and hist > 0 and spike > 0:
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
            elif adx_val > self.adx_threshold and mdi > pdi and hist < 0 and spike > 0:
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
                if side == 1: context.buy(add_lots)
                else: context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val):
        if self.position_scale >= MAX_SCALE: return False
        if self.bars_since_last_scale < 10: return False
        if self.direction == 1 and price < self.entry_price + atr_val: return False
        if self.direction == -1 and price > self.entry_price - atr_val: return False
        return True

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_stop_mult * atr_val * spec.multiplier
        if stop_dist <= 0: return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = context.close_raw * spec.multiplier * spec.margin_rate
        if margin <= 0: return 0
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
