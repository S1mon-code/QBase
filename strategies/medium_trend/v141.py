import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.supertrend import supertrend
from indicators.trend.ema import ema
from indicators.momentum.rsi import rsi
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV141(TimeSeriesStrategy):
    """
    策略简介：日线Supertrend趋势 + 4h EMA信号 + 5min RSI入场的三周期策略。

    使用指标：
    - Supertrend(10, 3.0) [日线]: 最大周期趋势方向
    - EMA(20) [4h]: 中周期趋势确认，close > EMA
    - RSI(14) [5min]: 小周期超卖回调精确入场
    - ATR(14) [5min]: 止损距离

    进场条件（做多）：日线ST dir=1 + 4h close>EMA + 5min RSI<30
    出场条件：ATR追踪止损, 分层止盈, ST翻转

    优点：三重过滤大幅降低假信号，入场精确
    缺点：三重条件同时满足机会较少
    """
    name = "medium_trend_v141"
    freq = "5min"
    warmup = 3000

    st_period: int = 10
    st_multiplier: float = 3.0
    ema_period: int = 20
    rsi_period: int = 14
    rsi_entry: float = 30.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._rsi = None
        self._atr = None
        self._avg_volume = None
        self._st_dir_daily = None
        self._ema_4h = None
        self._closes_4h = None
        self._daily_map = None
        self._4h_map = None

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

        # 5min indicators
        self._rsi = rsi(closes, self.rsi_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        # Aggregate to 4h (step=48)
        step_4h = 48
        n_4h = n // step_4h
        trim_4h = n_4h * step_4h
        closes_4h = closes[:trim_4h].reshape(n_4h, step_4h)[:, -1]
        self._ema_4h = ema(closes_4h, period=self.ema_period)
        self._closes_4h = closes_4h
        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step_4h - 1),
                                  n_4h - 1)

        # Aggregate to daily from 4h (step=4 from 4h perspective, but we use a bigger step)
        # Daily ~ 48*1 from 5min = 48 bars. Already 4h. Daily from 4h = ~1 bar for Chinese futures.
        # Use step=48*1=48 from 5min for daily directly
        step_d = 48  # same as 4h for Chinese futures (one session ~ 4h)
        # Actually daily should be larger. Let's use 4h bars to build daily:
        # Chinese futures: 4h ~ 1 per day. So daily = 4h directly.
        # Use the 4h data as daily proxy
        highs_4h = highs[:trim_4h].reshape(n_4h, step_4h).max(axis=1)
        lows_4h = lows[:trim_4h].reshape(n_4h, step_4h).min(axis=1)

        _, self._st_dir_daily = supertrend(
            highs_4h, lows_4h, closes_4h,
            period=self.st_period, multiplier=self.st_multiplier)
        self._daily_map = self._4h_map  # same mapping since daily ~ 4h for Chinese futures

    def on_bar(self, context):
        i = context.bar_index
        j4 = self._4h_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        rsi_val = self._rsi[i]
        atr_val = self._atr[i]
        st_dir = self._st_dir_daily[j4]
        ema_val = self._ema_4h[j4]
        close_4h = self._closes_4h[j4]
        if np.isnan(rsi_val) or np.isnan(atr_val) or np.isnan(st_dir) or np.isnan(ema_val):
            return

        daily_up = (st_dir == 1)
        mid_up = (close_4h > ema_val)
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

        if side == 1 and not daily_up:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and daily_up and mid_up and rsi_val < self.rsi_entry:
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
                    and daily_up and mid_up and rsi_val < 45):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
