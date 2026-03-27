import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.trend.linear_regression import linear_regression_slope, r_squared
from indicators.momentum.fisher_transform import fisher_transform
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV135(TimeSeriesStrategy):
    """
    策略简介：4h线性回归斜率+R平方趋势确认 + 5min Fisher Transform入场。

    使用指标：
    - Linear Regression Slope(20) + R-Squared(20) [4h]: 斜率>0且R2高=强趋势
    - Fisher Transform(10) [5min]: Fisher从负转正入场
    - ATR(14) [5min]: 止损距离

    进场条件（做多）：4h slope>0 & r2>0.5, 5min Fisher上穿trigger
    出场条件：ATR追踪止损, 分层止盈, slope转负或r2降低

    优点：R2量化趋势线性度，Fisher灵敏捕捉转折
    缺点：线性回归在非线性趋势中效果打折
    """
    name = "medium_trend_v135"
    freq = "5min"
    warmup = 2000

    lr_period: int = 20
    r2_threshold: float = 0.5
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._fisher = None
        self._fisher_trigger = None
        self._atr = None
        self._avg_volume = None
        self._slope_4h = None
        self._r2_4h = None
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

        self._fisher, self._fisher_trigger = fisher_transform(highs, lows, period=10)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 48
        n_4h = n // step
        trim = n_4h * step
        closes_4h = closes[:trim].reshape(n_4h, step)[:, -1]

        self._slope_4h = linear_regression_slope(closes_4h, period=self.lr_period)
        self._r2_4h = r_squared(closes_4h, period=self.lr_period)
        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1),
                                  len(self._slope_4h) - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        f_val = self._fisher[i]
        f_trig = self._fisher_trigger[i]
        atr_val = self._atr[i]
        slope = self._slope_4h[j]
        r2 = self._r2_4h[j]
        if np.isnan(f_val) or np.isnan(f_trig) or np.isnan(atr_val) or np.isnan(slope) or np.isnan(r2):
            return

        strong_uptrend = slope > 0 and r2 > self.r2_threshold
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

        if side == 1 and slope < 0:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and strong_uptrend and f_val > f_trig and f_val > 0:
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
                    and strong_uptrend):
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
