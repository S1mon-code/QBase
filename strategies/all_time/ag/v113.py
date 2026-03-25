import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.volatility.bollinger import bollinger_bands
from indicators.seasonality.weekday_effect import weekday_effect

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV113(TimeSeriesStrategy):
    """
    策略简介：Bollinger %B极端值 + 星期效应过滤的均值回归策略。

    使用指标：
    - Bollinger Bands(20, 2.0): 计算%B位置（0=下轨，1=上轨）
    - Weekday Effect(252): 历史星期收益模式，过滤不利日期
    - ATR(14): 止损距离计算

    进场条件（做多）：%B<0.05（价格触下轨）+ 当日星期效应z-score>0（历史上涨日）
    进场条件（做空）：%B>0.95（价格触上轨）+ 当日星期效应z-score<0（历史下跌日）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - %B回归中位（0.3-0.7之间）

    优点：双重过滤提高均值回归胜率，利用日历效应的统计优势
    缺点：星期效应可能周期性失效，强趋势中%B极端值持续
    """
    name = "ag_alltime_v113"
    warmup = 300
    freq = "daily"

    bb_period: int = 20           # Optuna: 15-30
    bb_std: float = 2.0           # Optuna: 1.5-2.5
    pctb_low: float = 0.05        # Optuna: 0.0-0.15
    pctb_high: float = 0.95       # Optuna: 0.85-1.0
    wd_lookback: int = 252        # Optuna: 120-504
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._bb_upper = None
        self._bb_mid = None
        self._bb_lower = None
        self._pctb = None
        self._wd_score = None
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
        n = len(closes)

        self._atr = atr(highs, lows, closes, period=14)
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            closes, period=self.bb_period, num_std=self.bb_std
        )

        # Compute %B: (close - lower) / (upper - lower)
        self._pctb = np.full(n, np.nan)
        for idx in range(n):
            bw = self._bb_upper[idx] - self._bb_lower[idx]
            if not np.isnan(bw) and bw > 1e-12:
                self._pctb[idx] = (closes[idx] - self._bb_lower[idx]) / bw

        # Weekday effect
        datetimes = context.get_full_datetime_array()
        self._wd_score, _ = weekday_effect(closes, datetimes, lookback=self.wd_lookback)

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
        if np.isnan(atr_val) or atr_val <= 0:
            return
        pctb = self._pctb[i]
        wd_score = self._wd_score[i]
        if np.isnan(pctb) or np.isnan(wd_score):
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

        # 3. Signal-based exit: %B returned to neutral
        if side == 1 and pctb > 0.7:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and pctb < 0.3:
            context.close_short()
            self._reset_state()
            return

        # Re-read position
        side, lots = context.position

        # 4. Entry: %B extreme + favorable weekday
        if side == 0:
            if pctb < self.pctb_low and wd_score > 0:
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
            elif pctb > self.pctb_high and wd_score < 0:
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
            signal_ok = (self.direction == 1 and pctb < 0.3) or \
                        (self.direction == -1 and pctb > 0.7)
            if signal_ok:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    if self.direction == 1:
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
