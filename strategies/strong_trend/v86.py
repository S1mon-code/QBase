"""
QBase Strong Trend Strategy v86 — Mean Reversion Speed (OU) + McGinley
========================================================================

策略简介：OU 过程的均值回归速度低值表示市场远离均值回归（趋势持续），
         McGinley Dynamic 均线确认方向。

使用指标：
  - Mean Reversion Speed / OU Speed (period=60): 低值 = 趋势持续
  - McGinley Dynamic (period=14): 自适应均线方向确认
  - ATR (period=14): trailing stop

进场条件：
  1. OU Speed < 0.05（低回归速度 = 趋势 regime）
  2. 收盘价 > McGinley（方向向上）
  3. McGinley 斜率 > 0

出场条件：
  1. ATR trailing stop（mult=4.0）
  2. OU Speed > 0.15（均值回归加速 = regime 切换）

优点：OU 模型有严格的数学基础，McGinley 自适应调整速度
缺点：OU 估计在小样本中不稳定
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.regime.mean_reversion_speed import ou_speed
from indicators.trend.mcginley import mcginley_dynamic
from indicators.volatility.atr import atr


class StrongTrendV86(TimeSeriesStrategy):
    """Low OU mean-reversion speed (trending) + McGinley direction."""
    name = "strong_trend_v86"
    warmup = 60
    freq = "daily"

    ou_threshold: float = 0.05
    mcg_period: int = 14
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._ou = None
        self._mcg = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._ou = ou_speed(closes, period=60)
        self._mcg = mcginley_dynamic(closes, period=self.mcg_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        ou_val = self._ou[i]
        mcg_val = self._mcg[i]
        atr_val = self._atr[i]
        if np.isnan(ou_val) or np.isnan(mcg_val) or np.isnan(atr_val) or i < 1:
            return

        mcg_prev = self._mcg[i - 1]
        if np.isnan(mcg_prev):
            return

        mcg_slope = mcg_val - mcg_prev

        # === Stop Loss Check ===
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # === Entry ===
        if side == 0 and ou_val < self.ou_threshold and price > mcg_val and mcg_slope > 0:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.stop_price = price - self.atr_trail_mult * atr_val
                self.highest_since_entry = price

        # === Signal Exit ===
        elif side == 1 and ou_val > 0.15:
            context.close_long()
            self._reset()

    def _calc_lots(self, context, price, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_trail_mult * atr_val * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = price * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        max_lots = int(context.equity * 0.30 / margin)
        return max(1, min(risk_lots, max_lots))

    def _reset(self):
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.stop_price = 0.0
