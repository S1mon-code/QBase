import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.ema import ema
from indicators.volume.obv import obv
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV101(TimeSeriesStrategy):
    """
    策略简介：EMA(20/60)金叉 + OBV量价趋势确认的日线多头策略。

    使用指标：
    - EMA(20, 60): 快慢均线交叉判断趋势
    - OBV: 累积量能趋势确认
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - EMA20 > EMA60（金叉后）
    - OBV在上升（当前OBV > 20日前OBV）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - EMA20 < EMA60（死叉）

    优点：日线级别信号稳定，OBV量价确认增加可靠性
    缺点：日线均线交叉滞后明显
    """
    name = "medium_trend_v101"
    warmup = 120
    freq = "daily"

    ema_fast: int = 20       # Optuna: 10-30
    ema_slow: int = 60       # Optuna: 40-80
    obv_lookback: int = 20   # Optuna: 10-40
    atr_stop_mult: float = 3.0  # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._ema_fast = None
        self._ema_slow = None
        self._obv = None
        self._atr = None
        self._avg_volume = None

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

        self._ema_fast = ema(closes, self.ema_fast)
        self._ema_slow = ema(closes, self.ema_slow)
        self._obv = obv(closes, volumes)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        ef = self._ema_fast[i]
        es = self._ema_slow[i]
        obv_val = self._obv[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(ef) or np.isnan(es) or np.isnan(obv_val):
            return

        obv_prev = self._obv[i - self.obv_lookback] if i >= self.obv_lookback else np.nan
        obv_rising = not np.isnan(obv_prev) and obv_val > obv_prev

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

        if side == 1 and ef < es:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and ef > es and obv_rising:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, ef, es):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, ef, es):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if ef <= es:
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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
