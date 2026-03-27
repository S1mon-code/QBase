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
from indicators.momentum.kama import kama
from indicators.trend.aroon import aroon
from indicators.volume.volume_spike import volume_spike

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV10(TimeSeriesStrategy):
    """
    策略简介：KAMA自适应均线 + Aroon振荡器 + 成交量异动确认的日线趋势策略。

    使用指标：
    - KAMA(10,2,30): 自适应均线，价格>KAMA=多头，斜率自适应波动
    - Aroon Oscillator(25): >0=上涨趋势，<0=下跌趋势，绝对值越大越强
    - Volume Spike(20,2.0): 放量确认突破有效性

    进场条件（做多）：价格>KAMA 且 KAMA斜率>0 且 Aroon振荡器>50 且 放量
    进场条件（做空）：价格<KAMA 且 KAMA斜率<0 且 Aroon振荡器<-50 且 放量

    出场条件：
    - ATR追踪止损
    - 分层止盈
    - KAMA斜率反转

    优点：KAMA自动适应市场波动，Aroon振荡器直观量化趋势强度
    缺点：KAMA在剧烈波动期可能过度平滑
    """
    name = "i_alltime_v10"
    warmup = 250
    freq = "daily"

    kama_period: int = 10
    aroon_period: int = 25
    aroon_osc_threshold: float = 50.0
    vol_spike_mult: float = 2.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._kama = None
        self._aroon_osc = None
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
        self._kama = kama(closes, self.kama_period, 2, 30)
        _, _, self._aroon_osc = aroon(highs, lows, period=self.aroon_period)
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
        kama_val = self._kama[i]
        aroon_osc = self._aroon_osc[i]
        spike = self._vol_spike[i]
        if np.isnan(kama_val) or np.isnan(aroon_osc) or np.isnan(spike):
            return

        # KAMA slope
        kama_slope = 0.0
        if i >= 1 and not np.isnan(self._kama[i - 1]):
            kama_slope = kama_val - self._kama[i - 1]

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
                if side == 1:
                    context.close_long(lots=cl)
                else:
                    context.close_short(lots=cl)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                cl = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=cl)
                else:
                    context.close_short(lots=cl)
                self._took_profit_3atr = True
                return

        # 3. Signal exit: KAMA slope reversal
        if side == 1 and kama_slope < 0 and price < kama_val:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and kama_slope > 0 and price > kama_val:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if price > kama_val and kama_slope > 0 and aroon_osc > self.aroon_osc_threshold and spike > 0:
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
            elif price < kama_val and kama_slope < 0 and aroon_osc < -self.aroon_osc_threshold and spike > 0:
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
