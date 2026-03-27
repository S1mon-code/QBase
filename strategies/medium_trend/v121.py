import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.trend.supertrend import supertrend
from indicators.momentum.rsi import rsi
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

_SPEC_MANAGER = ContractSpecManager()

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV121(TimeSeriesStrategy):
    """
    策略简介：日线Supertrend定方向 + 5min RSI超卖回调入场的多周期中趋势策略。

    使用指标：
    - Supertrend(10, 3.0) [日线]: 大周期趋势方向过滤
    - RSI(14) [5min]: 小周期超卖回调精确入场
    - ATR(14) [5min]: 止损距离计算

    进场条件（做多）：
    - 日线 Supertrend 方向 = 1（上升趋势）
    - 5min RSI < rsi_entry（超卖回调入场）

    出场条件：
    - ATR 追踪止损触发
    - 分层止盈（3ATR / 5ATR）
    - 日线 Supertrend 翻转为 -1

    优点：日线级别过滤假信号，5min级别精确捕捉回调入场点
    缺点：日线信号切换慢，可能在趋势末期仍持仓
    """
    name = "medium_trend_v121"
    freq = "5min"
    warmup = 2000

    st_period: int = 10
    st_multiplier: float = 3.0
    rsi_period: int = 14
    rsi_entry: float = 30.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._rsi = None
        self._atr = None
        self._avg_volume = None
        self._st_dir_daily = None
        self._daily_map = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        n = len(closes)

        self._rsi = rsi(closes, self.rsi_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 48  # 5min * 48 = 4h, approximate daily with 48*1=48 for 4h
        # For daily: 5min * 48 = 240min = 4h, daily ~= 4h*N
        # Chinese futures ~4h/day, so step=48 for daily
        n_daily = n // step
        trim = n_daily * step
        closes_d = closes[:trim].reshape(n_daily, step)[:, -1]
        highs_d = highs[:trim].reshape(n_daily, step).max(axis=1)
        lows_d = lows[:trim].reshape(n_daily, step).min(axis=1)

        _, self._st_dir_daily = supertrend(
            highs_d, lows_d, closes_d, period=self.st_period, multiplier=self.st_multiplier
        )
        self._daily_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1),
                                     len(self._st_dir_daily) - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._daily_map[i]
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        rsi_val = self._rsi[i]
        atr_val = self._atr[i]
        trend_dir = self._st_dir_daily[j]
        if np.isnan(rsi_val) or np.isnan(atr_val) or np.isnan(trend_dir):
            return

        self.bars_since_last_scale += 1

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

        if side == 1 and self.entry_price > 0:
            profit_atr = (price - self.entry_price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_long(lots=max(1, lots // 3))
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_long(lots=max(1, lots // 3))
                self._took_profit_3atr = True
                return

        if side == 1 and trend_dir != 1:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and trend_dir == 1 and rsi_val < self.rsi_entry:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and trend_dir == 1 and rsi_val < 40):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
