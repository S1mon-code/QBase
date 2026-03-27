import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.ml.kalman_trend import kalman_filter
from indicators.trend.tema import tema
from indicators.momentum.rsi import rsi
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV145(TimeSeriesStrategy):
    """
    策略简介：4h Kalman趋势 + 1h TEMA方向 + 10min RSI三周期策略。

    使用指标：
    - Kalman Filter [4h]: 自适应趋势线斜率
    - TEMA(20) [1h]: 中周期三重EMA方向确认
    - RSI(14) [10min]: 超卖入场
    - ATR(14) [10min]: 止损距离

    进场条件（做多）：4h Kalman上升 + 1h TEMA上升 + 10min RSI<30
    出场条件：ATR追踪止损, 分层止盈, Kalman转下降

    优点：Kalman自适应+TEMA快速响应双重确认
    缺点：Kalman对噪声初始估计敏感
    """
    name = "medium_trend_v145"
    freq = "10min"
    warmup = 3000

    tema_period: int = 20
    rsi_entry: float = 30.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._rsi = None
        self._atr = None
        self._avg_volume = None
        self._kalman_4h = None
        self._tema_1h = None
        self._4h_map = None
        self._1h_map = None

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

        self._rsi = rsi(closes, 14)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step_1h = 6
        n_1h = n // step_1h
        trim_1h = n_1h * step_1h
        closes_1h = closes[:trim_1h].reshape(n_1h, step_1h)[:, -1]
        self._tema_1h = tema(closes_1h, period=self.tema_period)
        self._1h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step_1h - 1), n_1h - 1)

        step_4h = 24
        n_4h = n // step_4h
        trim_4h = n_4h * step_4h
        closes_4h = closes[:trim_4h].reshape(n_4h, step_4h)[:, -1]
        self._kalman_4h = kalman_filter(closes_4h)
        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step_4h - 1), n_4h - 1)

    def on_bar(self, context):
        i = context.bar_index
        j4 = self._4h_map[i]
        j1 = self._1h_map[i]
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        rsi_val = self._rsi[i]
        atr_val = self._atr[i]
        k_cur = self._kalman_4h[j4]
        k_prev = self._kalman_4h[j4 - 1] if j4 > 0 else np.nan
        t_cur = self._tema_1h[j1]
        t_prev = self._tema_1h[j1 - 1] if j1 > 0 else np.nan
        if np.isnan(rsi_val) or np.isnan(atr_val) or np.isnan(k_cur) or np.isnan(k_prev) or np.isnan(t_cur) or np.isnan(t_prev):
            return

        kalman_up = k_cur > k_prev
        tema_up = t_cur > t_prev
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

        if side == 1 and not kalman_up:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and kalman_up and tema_up and rsi_val < self.rsi_entry:
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
                    and kalman_up and tema_up):
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
