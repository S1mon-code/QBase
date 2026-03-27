import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.ml.cusum_filter import cusum_event_filter
from indicators.trend.ema import ema
from indicators.volume.normalized_volume import normalized_volume
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV80(TimeSeriesStrategy):
    """
    策略简介：CUSUM Event Filter事件检测 + EMA趋势 + Normalized Volume的1h事件驱动策略。

    使用指标：
    - CUSUM Event Filter(1.0): 检测价格偏离累积超过阈值的事件
    - EMA(20): 趋势方向过滤，close>EMA为多头
    - Normalized Volume(20): >1.5为放量确认
    - ATR(14): 止损距离

    进场条件（做多）：CUSUM触发事件 且 close>EMA 且 Normalized Volume>1.5
    出场条件：追踪止损 / 分层止盈 / close < EMA

    优点：CUSUM对regime shift极其灵敏，结合EMA和Volume三重确认
    缺点：CUSUM阈值需精心调整，事件信号可能过稀
    """
    name = "medium_trend_v80"
    warmup = 400
    freq = "1h"

    cusum_threshold: float = 1.0      # Optuna: 0.5-2.0
    ema_period: int = 20              # Optuna: 10-40
    nv_threshold: float = 1.5         # Optuna: 1.0-2.5
    atr_stop_mult: float = 3.0       # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._cusum = None
        self._ema = None
        self._nv = None
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

        self._cusum = cusum_event_filter(closes, threshold=self.cusum_threshold)
        self._ema = ema(closes, self.ema_period)
        self._nv = normalized_volume(volumes, period=20)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        close_val = context.get_full_close_array()[i]
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        cusum_val = self._cusum[i]
        ema_val = self._ema[i]
        nv_val = self._nv[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(ema_val) or np.isnan(nv_val):
            return
        if np.isnan(cusum_val):
            cusum_val = 0

        above_ema = close_val > ema_val
        cusum_event = (cusum_val == 1)
        high_volume = (nv_val > self.nv_threshold)

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

        if side == 1 and not above_ema:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and cusum_event and above_ema and high_volume:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, above_ema):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, above_ema):
        if self.position_scale >= MAX_SCALE or self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        return above_ema

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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
