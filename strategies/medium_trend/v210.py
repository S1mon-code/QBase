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
from indicators.volume.chaikin_oscillator import chaikin_oscillator
from indicators.structure.oi_breakout import oi_breakout

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class MediumTrendV210(TimeSeriesStrategy):
    """
    策略简介：Chaikin Oscillator + OI Breakout的日线做多策略。

    使用指标：
    - Chaikin Oscillator(3, 10): A/D线的MACD，衡量资金流加速度
    - OI Breakout(20, 2.0): 持仓量突破，标志新趋势启动
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Chaikin Oscillator > 0 且上升（资金加速流入）
    - OI Breakout发出信号（持仓量放大确认）

    出场条件：
    - ATR追踪止损（3.5倍ATR）
    - 分层止盈（3ATR/5ATR）
    - Chaikin Oscillator < 0 持续3日

    优点：OI突破标志新仓位涌入，Chaikin验证资金方向
    缺点：OI突破可能是套保引起，不一定代表方向性建仓
    """
    name = "mt_v210"
    warmup = 60
    freq = "daily"

    atr_stop_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._chaikin = None
        self._oi_brk = None
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
        self.chaikin_neg_count = 0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()
        self._atr = atr(highs, lows, closes, period=14)
        self._chaikin = chaikin_oscillator(highs, lows, closes, volumes)
        self._oi_brk = oi_breakout(oi)
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
        ch_val = self._chaikin[i]
        oi_b = self._oi_brk[i]
        if np.isnan(ch_val) or np.isnan(oi_b):
            return
        prev_ch = self._chaikin[i - 1] if i > 0 else np.nan
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
            if ch_val < 0:
                self.chaikin_neg_count += 1
            else:
                self.chaikin_neg_count = 0
            if self.chaikin_neg_count >= 3:
                context.close_long()
                self._reset_state()
                return

        if side == 0 and ch_val > 0 and not np.isnan(prev_ch) and ch_val > prev_ch and oi_b > 0:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0
                self.chaikin_neg_count = 0
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
        self.chaikin_neg_count = 0
