import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.regime.trend_strength_composite import trend_strength
from indicators.momentum.cci import cci
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV125(TimeSeriesStrategy):
    """
    策略简介：4h趋势强度复合指标定方向 + 1h CCI突破入场的多周期策略。

    使用指标：
    - Trend Strength Composite(20) [4h]: 综合趋势强度评分0-100
    - CCI(20) [1h]: 顺势突破入场信号
    - ATR(14) [1h]: 止损距离计算

    进场条件（做多）：
    - 4h Trend Strength > ts_threshold（强趋势确认）
    - 1h CCI 上穿 cci_entry（动量突破）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR / 5ATR）
    - Trend Strength 跌破阈值

    优点：趋势强度综合评分比单一指标更稳健
    缺点：CCI在震荡市频繁假突破
    """
    name = "medium_trend_v125"
    freq = "1h"
    warmup = 500

    ts_period: int = 20
    ts_threshold: float = 60.0
    cci_period: int = 20
    cci_entry: float = 100.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._cci = None
        self._atr = None
        self._avg_volume = None
        self._ts_4h = None
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

        self._cci = cci(highs, lows, closes, period=self.cci_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 4  # 1h * 4 = 4h
        n_4h = n // step
        trim = n_4h * step
        closes_4h = closes[:trim].reshape(n_4h, step)[:, -1]
        highs_4h = highs[:trim].reshape(n_4h, step).max(axis=1)
        lows_4h = lows[:trim].reshape(n_4h, step).min(axis=1)

        self._ts_4h = trend_strength(closes_4h, highs_4h, lows_4h, period=self.ts_period)
        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1),
                                  len(self._ts_4h) - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        cci_val = self._cci[i]
        atr_val = self._atr[i]
        ts_val = self._ts_4h[j]
        if np.isnan(cci_val) or np.isnan(atr_val) or np.isnan(ts_val):
            return

        prev_cci = self._cci[i - 1] if i > 0 else np.nan
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

        if side == 1 and ts_val < self.ts_threshold * 0.6:
            context.close_long()
            self._reset_state()
            return

        if (side == 0 and ts_val > self.ts_threshold
                and not np.isnan(prev_cci) and prev_cci <= self.cci_entry and cci_val > self.cci_entry):
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
                    and ts_val > self.ts_threshold):
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
