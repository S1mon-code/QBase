import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.kalman_trend import kalman_filter
from indicators.momentum.cci import cci
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV134(TimeSeriesStrategy):
    """
    策略简介：日线Kalman滤波趋势 + 30min CCI回调入场的多周期策略。

    使用指标：
    - Kalman Filter [日线]: 自适应趋势线，斜率判断方向
    - CCI(20) [30min]: 回调至超卖区域入场
    - ATR(14) [30min]: 止损距离

    进场条件（做多）：日线Kalman上升, 30min CCI < -100（超卖回调）
    出场条件：ATR追踪止损, 分层止盈, Kalman转为下降

    优点：Kalman自适应噪声过滤，CCI捕捉回调
    缺点：Kalman参数敏感度高
    """
    name = "medium_trend_v134"
    freq = "30min"
    warmup = 1000

    cci_period: int = 20
    cci_entry: float = -100.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._cci = None
        self._atr = None
        self._avg_volume = None
        self._kalman_d = None
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

        self._cci = cci(highs, lows, closes, period=self.cci_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 8  # 30min * 8 = 4h ~ daily
        n_d = n // step
        trim = n_d * step
        closes_d = closes[:trim].reshape(n_d, step)[:, -1]

        self._kalman_d = kalman_filter(closes_d)
        self._daily_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1),
                                     len(self._kalman_d) - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._daily_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        cci_val = self._cci[i]
        atr_val = self._atr[i]
        k_cur = self._kalman_d[j]
        k_prev = self._kalman_d[j - 1] if j > 0 else np.nan
        if np.isnan(cci_val) or np.isnan(atr_val) or np.isnan(k_cur) or np.isnan(k_prev):
            return

        kalman_rising = k_cur > k_prev
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

        if side == 1 and not kalman_rising:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and kalman_rising and cci_val < self.cci_entry:
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
                    and kalman_rising):
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
