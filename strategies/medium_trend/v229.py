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
from indicators.volatility.adr import average_day_range
from indicators.momentum.awesome_oscillator import ao

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class MediumTrendV229(TimeSeriesStrategy):
    """
    策略简介：ADR日均波幅低位 + AO振荡器方向的1h做多策略。

    使用指标：
    - ADR(14): 平均日波幅，低位=蓄势
    - Awesome Oscillator(5, 34): 快慢中价差，方向指示
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - ADR处于近20期低30%（波幅收缩）
    - AO > 0 且 AO上升（动量向上加速）

    出场条件：
    - ATR追踪止损（3.0倍ATR）
    - 分层止盈
    - AO < 0（动量转负）

    优点：ADR低位入场时止损距离小，AO简洁有效
    缺点：ADR低位可能是流动性枯竭而非蓄势
    """
    name = "mt_v229"
    warmup = 60
    freq = "1h"

    adr_lookback: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._adr = None
        self._ao = None
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
        self._atr = atr(highs, lows, closes, period=14)
        self._adr = average_day_range(highs, lows)
        self._ao = ao(highs, lows)
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
        adr_val = self._adr[i]
        ao_val = self._ao[i]
        if np.isnan(adr_val) or np.isnan(ao_val):
            return
        prev_ao = self._ao[i - 1] if i > 0 else np.nan
        if np.isnan(prev_ao):
            return
        self.bars_since_last_scale += 1

        adr_low = False
        if i >= self.adr_lookback:
            adr_win = self._adr[i - self.adr_lookback:i]
            valid = adr_win[~np.isnan(adr_win)]
            if len(valid) > 5:
                adr_low = adr_val <= np.percentile(valid, 30)

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

        if side == 1 and ao_val < 0:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and adr_low and ao_val > 0 and ao_val > prev_ao:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0
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
