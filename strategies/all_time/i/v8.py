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
from indicators.trend.linear_regression import linear_regression
from indicators.volatility.chop import choppiness
from indicators.volume.obv import obv

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV8(TimeSeriesStrategy):
    """
    策略简介：线性回归斜率定方向 + Choppiness过滤震荡 + OBV量价确认的日线策略。

    使用指标：
    - Linear Regression(20): 斜率>0=上涨趋势，R²>0.5=趋势可靠
    - Choppiness(14): <38.2=强趋势，>61.8=震荡（只在趋势状态交易）
    - OBV: 成交量加权价格方向，斜率确认资金流向

    进场条件（做多）：LR斜率>0 且 R²>0.5 且 Choppiness<50 且 OBV上升
    进场条件（做空）：LR斜率<0 且 R²>0.5 且 Choppiness<50 且 OBV下降

    出场条件：
    - ATR追踪止损
    - 分层止盈
    - Choppiness>61.8（进入震荡，趋势结束）

    优点：线性回归+R²组合能精确量化趋势强度，Choppiness避开震荡
    缺点：LR对异常值敏感，Choppiness滞后于趋势转换
    """
    name = "i_alltime_v8"
    warmup = 250
    freq = "daily"

    lr_period: int = 20
    chop_period: int = 14
    chop_threshold: float = 50.0
    r2_threshold: float = 0.5
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._lr_slope = None
        self._lr_r2 = None
        self._chop = None
        self._obv = None
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
        _, self._lr_slope, self._lr_r2 = linear_regression(closes, self.lr_period)
        self._chop = choppiness(highs, lows, closes, self.chop_period)
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
        slope = self._lr_slope[i]
        r2 = self._lr_r2[i]
        chop = self._chop[i]
        obv_val = self._obv[i]
        if np.isnan(slope) or np.isnan(r2) or np.isnan(chop) or np.isnan(obv_val):
            return

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
                if side == 1:
                    context.close_long(lots=cl)
                else:
                    context.close_short(lots=cl)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                cl = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=cl)
                else:
                    context.close_short(lots=cl)
                self._took_profit_3atr = True
                return

        # 3. Signal exit: Choppiness goes high (market becomes choppy)
        if side != 0 and chop > 61.8:
            if side == 1:
                context.close_long()
            else:
                context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if slope > 0 and r2 > self.r2_threshold and chop < self.chop_threshold and obv_slope > 0:
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
            elif slope < 0 and r2 > self.r2_threshold and chop < self.chop_threshold and obv_slope < 0:
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
                if side == 1:
                    context.buy(add_lots)
                else:
                    context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1 and price < self.entry_price + atr_val:
            return False
        if self.direction == -1 and price > self.entry_price - atr_val:
            return False
        return True

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
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.direction = 0
