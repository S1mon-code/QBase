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
from indicators.trend.ema import ema
from indicators.momentum.rsi import rsi
from indicators.volume.force_index import force_index

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV4(TimeSeriesStrategy):
    """
    策略简介：EMA双均线交叉 + RSI过滤 + Force Index量价确认的日线趋势策略。

    使用指标：
    - EMA(20/60): 快慢均线交叉定方向
    - RSI(14): 超买超卖过滤，避免追高/追低
    - Force Index(13): 量价合力，正=多头力量，负=空头力量

    进场条件（做多）：EMA20>EMA60 且 RSI<70 且 Force Index>0
    进场条件（做空）：EMA20<EMA60 且 RSI>30 且 Force Index<0

    出场条件：
    - ATR追踪止损
    - 分层止盈
    - EMA交叉反转

    优点：经典双均线系统配合RSI过滤极端区域，Force Index确认量价一致
    缺点：均线交叉信号滞后，震荡市频繁假交叉
    """
    name = "i_alltime_v4"
    warmup = 250
    freq = "daily"

    ema_fast: int = 20
    ema_slow: int = 60
    rsi_period: int = 14
    fi_period: int = 13
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._ema_fast = None
        self._ema_slow = None
        self._rsi = None
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
        self._ema_fast = ema(closes, self.ema_fast)
        self._ema_slow = ema(closes, self.ema_slow)
        self._rsi = rsi(closes, self.rsi_period)
        self._fi = force_index(closes, volumes, self.fi_period)

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
        ef = self._ema_fast[i]
        es = self._ema_slow[i]
        rsi_val = self._rsi[i]
        fi_val = self._fi[i]
        if np.isnan(ef) or np.isnan(es) or np.isnan(rsi_val) or np.isnan(fi_val):
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

        # 3. Signal exit: EMA crossover reversal
        if side == 1 and ef < es:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and ef > es:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if ef > es and rsi_val < 70 and fi_val > 0:
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
            elif ef < es and rsi_val > 30 and fi_val < 0:
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
