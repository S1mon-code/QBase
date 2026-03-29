import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.volatility.atr import atr
from indicators.momentum.ultimate_oscillator import ultimate_oscillator
from indicators.trend.dema import dema

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class MediumTrendV236(TimeSeriesStrategy):
    """
    策略简介：Ultimate Oscillator多周期 + DEMA双指数均线的1h做多策略。

    使用指标：
    - Ultimate Oscillator(7, 14, 28): 三周期加权买压振荡器
    - DEMA(20): 双指数均线，低延迟
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - UO从 < 30回升至 > 30（超卖回升）
    - 价格 > DEMA（趋势方向向上）

    出场条件：
    - ATR追踪止损（3.5倍ATR）
    - 分层止盈
    - UO > 70后回落至 < 50（超买衰竭）

    优点：UO融合三周期减少假信号，DEMA快速响应
    缺点：UO超卖区间可能持续较长
    """
    name = "mt_v236"
    warmup = 60
    freq = "1h"

    dema_period: int = 20
    uo_entry: float = 30.0
    atr_stop_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._uo = None
        self._dema = None
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
        self.uo_was_high = False

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        self._atr = atr(highs, lows, closes, period=14)
        self._uo = ultimate_oscillator(highs, lows, closes)
        self._dema = dema(closes, self.dema_period)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return
        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        uo_val = self._uo[i]
        dema_val = self._dema[i]
        if np.isnan(uo_val) or np.isnan(dema_val):
            return
        prev_uo = self._uo[i - 1] if i > 0 else np.nan
        if np.isnan(prev_uo):
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

        if side == 1:
            if uo_val > 70:
                self.uo_was_high = True
            if self.uo_was_high and uo_val < 50:
                context.close_long()
                self._reset_state()
                return

        if side == 0 and prev_uo < self.uo_entry and uo_val > self.uo_entry and price > dema_val:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0
                self.uo_was_high = False
        elif side == 1 and self.position_scale < MAX_SCALE and self.bars_since_last_scale >= 10 and price > self.entry_price + atr_val:
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
        self.uo_was_high = False
