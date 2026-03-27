import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.i.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.volatility.atr import atr
from indicators.trend.zlema import zlema
from indicators.momentum.ppo import ppo
from indicators.volume.obv import obv

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV20(TimeSeriesStrategy):
    """
    策略简介：ZLEMA零延迟均线 + PPO标准化动量 + OBV量价确认的4h趋势策略。

    使用指标：
    - ZLEMA(20): 零延迟指数均线，修正EMA的滞后
    - PPO(12,26,9): 百分比振荡器
    - OBV: 累积量价方向

    进场条件（做多）：价格>ZLEMA 且 斜率>0 且 PPO线>信号线 且 OBV上升
    进场条件（做空）：价格<ZLEMA 且 斜率<0 且 PPO线<信号线 且 OBV下降

    出场条件：ATR追踪止损 / 分层止盈 / 价格穿越ZLEMA+PPO确认

    优点：ZLEMA修正滞后问题，OBV确认资金持续流入/流出
    缺点：零延迟修正可能在窄幅震荡中过度敏感
    """
    name = "i_alltime_v20"
    warmup = 400
    freq = "4h"

    zlema_period: int = 20
    ppo_fast: int = 12
    ppo_slow: int = 26
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
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
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        self._zlema = zlema(closes, self.zlema_period)
        self._ppo_line, self._ppo_signal, _ = ppo(closes, self.ppo_fast, self.ppo_slow, 9)
        self._obv = obv(closes, volumes)

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

        zl = self._zlema[i]
        ppo_l = self._ppo_line[i]
        ppo_s = self._ppo_signal[i]
        obv_val = self._obv[i]
        if np.isnan(zl) or np.isnan(ppo_l) or np.isnan(ppo_s) or np.isnan(obv_val):
            return
        zl_slope = 0.0
        if i >= 1 and not np.isnan(self._zlema[i - 1]):
            zl_slope = zl - self._zlema[i - 1]
        obv_slope = 0.0
        if i >= 5 and not np.isnan(self._obv[i - 5]):
            obv_slope = obv_val - self._obv[i - 5]

        self.bars_since_last_scale += 1

        # 1. Stop loss
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

        # 2. Tiered profit-taking
        if side != 0 and self.entry_price > 0:
            profit_atr = ((price - self.entry_price) / atr_val) if side == 1 else ((self.entry_price - price) / atr_val)
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                cl = max(1, lots // 3)
                if side == 1: context.close_long(lots=cl)
                else: context.close_short(lots=cl)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                cl = max(1, lots // 3)
                if side == 1: context.close_long(lots=cl)
                else: context.close_short(lots=cl)
                self._took_profit_3atr = True
                return

        # 3. Signal-based exit
        if side == 1 and (price < zl and ppo_l < ppo_s):
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and (price > zl and ppo_l > ppo_s):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if price > zl and zl_slope > 0 and ppo_l > ppo_s and obv_slope > 0:
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
            elif price < zl and zl_slope < 0 and ppo_l < ppo_s and obv_slope < 0:
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

        # 5. Scale-in
        elif side != 0 and self._should_add(price, atr_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if side == 1: context.buy(add_lots)
                else: context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val):
        if self.position_scale >= MAX_SCALE: return False
        if self.bars_since_last_scale < 10: return False
        if self.direction == 1 and price < self.entry_price + atr_val: return False
        if self.direction == -1 and price > self.entry_price - atr_val: return False
        return True

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_stop_mult * atr_val * spec.multiplier
        if stop_dist <= 0: return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = context.close_raw * spec.multiplier * spec.margin_rate
        if margin <= 0: return 0
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
