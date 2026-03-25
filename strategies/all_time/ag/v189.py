import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.momentum.stochastic import stochastic
from indicators.trend.keltner import keltner
from indicators.seasonality.weekday_effect import weekday_effect
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV189(TimeSeriesStrategy):
    """
    策略简介：Stochastic在Keltner通道极端区域 + 有利星期几效应的均值回归策略。

    使用指标：
    - Stochastic(14,3): 超买超卖振荡器
    - Keltner Channel(20,10,1.5): 动态波动通道，价格触及上下轨为极端
    - Weekday Effect(252): 星期几效应，在历史上有利的交易日入场
    - ATR(14): 止损距离计算

    进场条件（做多）：%K<20（超卖）+ 价格触及Keltner下轨 + weekday_score>0
    进场条件（做空）：%K>80（超买）+ 价格触及Keltner上轨 + weekday_score<0

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - Stochastic回到中性区域（30-70）

    优点：多维度极端确认+日历效应增加边际优势
    缺点：强趋势中超买超卖信号持续失效
    """
    name = "ag_alltime_v189"
    warmup = 500
    freq = "1h"

    stoch_k: int = 14             # Optuna: 9-21
    stoch_d: int = 3              # Optuna: 3-5
    kelt_ema: int = 20            # Optuna: 14-30
    kelt_atr: int = 10            # Optuna: 7-14
    kelt_mult: float = 1.5       # Optuna: 1.0-2.5
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._stoch_k = None
        self._stoch_d = None
        self._kelt_upper = None
        self._kelt_mid = None
        self._kelt_lower = None
        self._weekday_score = None
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
        datetimes = context.get_full_datetime_array()

        self._stoch_k, self._stoch_d = stochastic(
            highs, lows, closes, self.stoch_k, self.stoch_d)
        self._kelt_upper, self._kelt_mid, self._kelt_lower = keltner(
            highs, lows, closes, self.kelt_ema, self.kelt_atr, self.kelt_mult)
        self._weekday_score, _ = weekday_effect(closes, datetimes, lookback=252)
        self._atr = atr(highs, lows, closes, period=14)

        window = 20
        self._avg_volume = np.full_like(volumes, np.nan)
        for idx in range(window, len(volumes)):
            self._avg_volume[idx] = np.mean(volumes[idx - window:idx])

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        sk = self._stoch_k[i]
        ku = self._kelt_upper[i]
        kl = self._kelt_lower[i]
        ws = self._weekday_score[i]

        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(sk) or np.isnan(ku) or np.isnan(kl):
            return
        if np.isnan(ws):
            ws = 0.0  # neutral if no weekday data

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
            if side == 1:
                profit_atr = (price - self.entry_price) / atr_val
            else:
                profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_3atr = True
                return

        # 3. Signal exit: Stochastic returns to neutral
        if side == 1 and sk > 70:
            context.close_long()
            self._reset_state()
        elif side == -1 and sk < 30:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry
        if side == 0:
            if sk < 20 and price <= kl and ws >= 0:
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
            elif sk > 80 and price >= ku and ws <= 0:
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
            if self.direction == 1 and sk < 40:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and sk > 60:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
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
        from alphaforge.data.contract_specs import ContractSpecManager
        spec = ContractSpecManager().get(context.symbol)
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
