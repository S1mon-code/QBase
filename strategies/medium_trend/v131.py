import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.donchian import donchian
from indicators.momentum.williams_r import williams_r
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV131(TimeSeriesStrategy):
    """
    策略简介：日线Donchian通道突破方向 + 1h Williams %R入场的多周期策略。

    使用指标：
    - Donchian(20) [日线]: 价格在上轨附近=上升趋势
    - Williams %R(14) [1h]: 超卖回调入场
    - ATR(14) [1h]: 止损距离

    进场条件（做多）：日线close > Donchian mid, 1h %R < -80（超卖）
    出场条件：ATR追踪止损, 分层止盈, close < Donchian lower

    优点：Donchian通道简洁有效，%R灵敏
    缺点：Donchian在窄幅盘整时通道很窄，假突破多
    """
    name = "medium_trend_v131"
    freq = "1h"
    warmup = 500

    donchian_period: int = 20
    wr_period: int = 14
    wr_entry: float = -80.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._wr = None
        self._atr = None
        self._avg_volume = None
        self._dc_upper_d = None
        self._dc_lower_d = None
        self._dc_mid_d = None
        self._closes_d = None
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

        self._wr = williams_r(highs, lows, closes, period=self.wr_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 4  # 1h * 4 = 4h ~ daily approx
        n_d = n // step
        trim = n_d * step
        closes_d = closes[:trim].reshape(n_d, step)[:, -1]
        highs_d = highs[:trim].reshape(n_d, step).max(axis=1)
        lows_d = lows[:trim].reshape(n_d, step).min(axis=1)

        self._dc_upper_d, self._dc_lower_d, self._dc_mid_d = donchian(
            highs_d, lows_d, period=self.donchian_period)
        self._closes_d = closes_d
        self._daily_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1),
                                     len(closes_d) - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._daily_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        wr_val = self._wr[i]
        atr_val = self._atr[i]
        dc_mid = self._dc_mid_d[j]
        dc_lower = self._dc_lower_d[j]
        cd = self._closes_d[j]
        if np.isnan(wr_val) or np.isnan(atr_val) or np.isnan(dc_mid):
            return

        uptrend = cd > dc_mid
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

        if side == 1 and not np.isnan(dc_lower) and cd < dc_lower:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and uptrend and wr_val < self.wr_entry:
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
                    and uptrend):
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
