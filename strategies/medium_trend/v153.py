import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.linear_regression import linear_regression_slope
from indicators.trend.keltner import keltner
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV153(TimeSeriesStrategy):
    """
    策略简介：日线LR Slope + 4h Keltner通道 + 30min ROC三周期策略。

    使用指标：
    - Linear Regression Slope(20) [日线]: 斜率>0确认大趋势
    - Keltner(20,10,1.5) [4h]: 价格在中轨上方确认中周期
    - ROC(12) [30min]: 动量为正入场
    - ATR(14) [30min]: 止损距离

    进场条件（做多）：日线slope>0 + 4h close>Keltner mid + 30min ROC>0
    出场条件：ATR追踪止损, 分层止盈, slope转负

    优点：线性回归量化趋势斜率，Keltner波动率自适应
    缺点：线性回归在非线性行情中效果打折
    """
    name = "medium_trend_v153"
    freq = "30min"
    warmup = 1500

    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._roc = None
        self._atr = None
        self._avg_volume = None
        self._slope_d = None
        self._kc_mid_4h = None
        self._closes_4h = None
        self._d_map = None
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

        self._roc = rate_of_change(closes, 12)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step_4h = 8
        n_4h = n // step_4h
        trim_4h = n_4h * step_4h
        closes_4h = closes[:trim_4h].reshape(n_4h, step_4h)[:, -1]
        highs_4h = highs[:trim_4h].reshape(n_4h, step_4h).max(axis=1)
        lows_4h = lows[:trim_4h].reshape(n_4h, step_4h).min(axis=1)
        _, self._kc_mid_4h, _ = keltner(highs_4h, lows_4h, closes_4h, ema=20, atr=10, mult=1.5)
        self._closes_4h = closes_4h
        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step_4h - 1), n_4h - 1)

        self._slope_d = linear_regression_slope(closes_4h, period=20)
        self._d_map = self._4h_map

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        roc_val = self._roc[i]
        atr_val = self._atr[i]
        slope = self._slope_d[j]
        kc_mid = self._kc_mid_4h[j]
        c4h = self._closes_4h[j]
        if np.isnan(roc_val) or np.isnan(atr_val) or np.isnan(slope) or np.isnan(kc_mid):
            return

        slope_up = slope > 0
        above_mid = c4h > kc_mid
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

        if side == 1 and not slope_up:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and slope_up and above_mid and roc_val > 0:
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
                    and slope_up and above_mid):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
