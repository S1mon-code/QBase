import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.trend.linear_regression import linear_regression_slope, r_squared
from indicators.volume.klinger import klinger
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV55(TimeSeriesStrategy):
    """
    策略简介：Linear Regression斜率 + R-Squared + Klinger量价的30min回归趋势策略。

    使用指标：
    - Linear Regression Slope(20): 斜率>0且R^2>0.5表示强线性上升趋势
    - R-Squared(20): 拟合优度，>0.5时趋势可靠
    - Klinger(34, 55, 13): KVO>signal为多头量价信号
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - LR Slope > 0（向上趋势）
    - R^2 > r2_threshold（趋势线性度高）
    - KVO > KVO signal（量价支持）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - LR Slope < 0（趋势反转）

    优点：R^2过滤非线性噪音，Klinger量价双确认
    缺点：线性回归假设趋势直线，无法捕捉非线性加速
    """
    name = "medium_trend_v55"
    warmup = 600
    freq = "30min"

    lr_period: int = 20               # Optuna: 10-40
    r2_threshold: float = 0.5         # Optuna: 0.3-0.7
    atr_stop_mult: float = 3.0       # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._lr_slope = None
        self._r2 = None
        self._kvo = None
        self._kvo_sig = None
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

        self._lr_slope = linear_regression_slope(closes, self.lr_period)
        self._r2 = r_squared(closes, self.lr_period)
        self._kvo, self._kvo_sig = klinger(highs, lows, closes, volumes)
        self._atr = atr(highs, lows, closes, period=14)
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
        slope = self._lr_slope[i]
        r2 = self._r2[i]
        kvo = self._kvo[i]
        kvo_s = self._kvo_sig[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(slope) or np.isnan(r2) or np.isnan(kvo) or np.isnan(kvo_s):
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

        if side == 1 and slope < 0:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and slope > 0 and r2 > self.r2_threshold and kvo > kvo_s:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, slope, r2):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, slope, r2):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if slope <= 0 or r2 < self.r2_threshold:
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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
