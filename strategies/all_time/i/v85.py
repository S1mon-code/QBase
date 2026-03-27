import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.i.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.volatility.range_expansion import range_expansion
from indicators.trend.aroon import aroon
from indicators.volume.cmf import cmf
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV85(TimeSeriesStrategy):
    """
    策略简介：Range Expansion波动扩张 + Aroon(25)趋势确认 + CMF资金流向的突破策略。

    使用指标：
    - Range Expansion: 检测价格区间扩张，标志突破开始
    - Aroon(25): up>down看多，down>up看空，确认趋势方向
    - CMF(20): Chaikin资金流向，>0多头<0空头
    - ATR(14): 止损距离计算

    进场条件（做多）：Range Expansion触发 + Aroon Up > Aroon Down + CMF > 0
    进场条件（做空）：Range Expansion触发 + Aroon Down > Aroon Up + CMF < 0

    出场条件：ATR追踪止损 / 分层止盈 / Aroon方向反转

    优点：Range Expansion精确捕捉突破起点，Aroon确认方向持续性
    缺点：Range Expansion信号频繁，需要Aroon和CMF严格过滤
    """
    name = "i_alltime_v85"
    warmup = 250
    freq = "daily"

    aroon_period: int = 25
    cmf_period: int = 20
    aroon_diff: float = 30.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._re = None
        self._aroon_up = None
        self._aroon_down = None
        self._cmf = None
        self._atr = None
        self._avg_volume = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.direction = 0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._re = range_expansion(highs, lows, closes)
        ar_up, ar_down, _ = aroon(highs, lows, period=self.aroon_period)
        self._aroon_up = ar_up
        self._aroon_down = ar_down
        self._cmf = cmf(highs, lows, closes, volumes, period=self.cmf_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        re_val = self._re[i]
        ar_up = self._aroon_up[i]
        ar_down = self._aroon_down[i]
        cmf_val = self._cmf[i]
        if np.isnan(atr_val) or np.isnan(ar_up) or np.isnan(ar_down) or np.isnan(cmf_val):
            return

        self.bars_since_last_scale += 1

        aroon_bullish = (ar_up - ar_down) > self.aroon_diff
        aroon_bearish = (ar_down - ar_up) > self.aroon_diff

        # ── 1. 止损 ──
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return
        elif side == -1:
            self.lowest_since_entry = min(self.lowest_since_entry, price)
            trailing = self.lowest_since_entry + self.atr_stop_mult * atr_val
            self.stop_price = min(self.stop_price, trailing)
            if price >= self.stop_price:
                context.close_short()
                self._reset_state()
                return

        # ── 2. 分层止盈 ──
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
        elif side == -1 and self.entry_price > 0:
            profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_3atr = True
                return

        # ── 3. 信号退出：Aroon方向反转 ──
        if side == 1 and aroon_bearish:
            context.close_long()
            self._reset_state()
        elif side == -1 and aroon_bullish:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场 ──
        if side == 0 and re_val:
            if aroon_bullish and cmf_val > 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
                    self.direction = 1
            elif aroon_bearish and cmf_val < 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
                    self.direction = -1

        # ── 5. 加仓 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and aroon_bullish):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and aroon_bearish):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.sell(add)
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
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.direction = 0
