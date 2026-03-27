import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.kalman_trend import kalman_filter
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV3(TimeSeriesStrategy):
    """
    策略简介：Kalman Filter趋势 + ATR扩张确认的做多策略（5min频率）。

    使用指标：
    - Kalman Filter: 自适应趋势跟踪，价格在Kalman线上方为多头
    - ATR(14): 波动扩张确认趋势启动 + 止损距离

    进场条件（做多）：价格 > Kalman滤波值 且 ATR > ATR均值（波动扩张）
    出场条件：ATR追踪止损 / 分层止盈 / 价格跌破Kalman线

    优点：Kalman自适应平滑，噪音过滤优于固定均线
    缺点：Kalman参数敏感，process_noise需要调优
    """
    name = "mt_v3"
    warmup = 2000
    freq = "5min"

    process_noise: float = 0.01
    measurement_noise: float = 1.0
    atr_expansion_mult: float = 1.2
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._kalman = None
        self._atr = None
        self._atr_ma = None
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

        self._kalman = kalman_filter(closes, process_noise=self.process_noise,
                                     measurement_noise=self.measurement_noise)
        self._atr = atr(highs, lows, closes, period=14)
        # Rolling mean of ATR for expansion detection
        n = len(closes)
        atr_cumsum = np.cumsum(np.insert(self._atr, 0, 0.0))
        self._atr_ma = np.full(n, np.nan)
        w = 50
        self._atr_ma[w:] = (atr_cumsum[w:n] - atr_cumsum[:n - w]) / w
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
        kalman_val = self._kalman[i]
        atr_ma = self._atr_ma[i]
        if np.isnan(atr_val) or np.isnan(kalman_val) or np.isnan(atr_ma):
            return

        self.bars_since_last_scale += 1
        above_kalman = price > kalman_val
        atr_expanding = atr_val > atr_ma * self.atr_expansion_mult

        # 1. Stop loss
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

        # 2. Tiered profit-taking
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

        # 3. Signal exit: price below Kalman
        if side == 1 and not above_kalman:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and above_kalman and atr_expanding:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, above_kalman):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, above_kalman):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if not above_kalman:
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
