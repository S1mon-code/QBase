"""
Boss Strategy v7 — Linear Regression Quality Trend
====================================================
Only trade CLEAN trends. R² filter eliminates noisy/choppy trends.
LONG ONLY. Supports scale-in (0-3).

Usage:
    ./run.sh strategies/boss/v7.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.trend.linear_regression import linear_regression_slope, r_squared
from indicators.volume.obv import obv

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BossV7(TimeSeriesStrategy):
    """
    策略简介：线性回归斜率+R²双重过滤，只交易干净有序的趋势，拒绝噪音。
    交易哲学：不是所有趋势都值得交易。R²衡量趋势的"质量"——高R²意味着价格
              沿着干净的直线移动，低R²意味着虽然方向对但过程很颠簸。
              我们只要最干净的趋势。
    使用指标：
      - Linear Regression Slope(30): 趋势方向和强度
      - R²(30): 趋势质量/有序度（0=混沌, 1=完美直线）
      - OBV: 资金流向确认
      - ATR(14): 止损距离
    进场条件（做多）：
      1. LinReg Slope > 0（趋势向上）
      2. R² > r2_threshold（趋势足够干净有序）
      3. OBV 上升中（OBV[i] > OBV[i-10]，资金持续流入）
    出场条件：
      - R² < 0.3（趋势变得混乱）
      - 或 Slope 转负（方向反转）
      - ATR追踪止损
      - 分层止盈（3ATR / 5ATR）
    优点：R²过滤是独特的"趋势质量"维度，大幅减少震荡市假信号
    缺点：R²计算窗口较长，反应偏慢；趋势初期R²还不够高可能错过起点
    """
    name = "boss_v7"
    warmup = 150
    freq = "daily"

    # Tunable parameters (<=5)
    lr_period: int = 30
    r2_threshold: float = 0.5
    obv_lookback: int = 10
    atr_stop_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._slope = None
        self._r2 = None
        self._obv = None
        self._atr = None
        self._avg_vol = None

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

        self._slope = linear_regression_slope(closes, self.lr_period)
        self._r2 = r_squared(closes, self.lr_period)
        self._obv = obv(closes, volumes)
        self._atr = atr(highs, lows, closes, period=14)

        n = len(volumes)
        cumsum = np.cumsum(np.insert(volumes, 0, 0.0))
        self._avg_vol = np.full(n, np.nan)
        if n >= 20:
            self._avg_vol[20:] = (cumsum[21:n + 1] - cumsum[1:n - 19]) / 20

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_vol[i]) and context.volume < self._avg_vol[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        slope_val = self._slope[i]
        r2_val = self._r2[i]
        obv_val = self._obv[i]
        if np.isnan(slope_val) or np.isnan(r2_val) or np.isnan(obv_val):
            return

        self.bars_since_last_scale += 1

        # 1. STOP LOSS
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

        # 2. TIERED PROFIT-TAKING
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

        # 3. SIGNAL EXIT: R² drops below 0.3 OR slope turns negative
        if side == 1:
            if r2_val < 0.3 or slope_val < 0:
                context.close_long()
                self._reset_state()
                return

        # 4. ENTRY: slope > 0 + R² above threshold + OBV rising
        if side == 0:
            if slope_val <= 0:
                return
            if r2_val < self.r2_threshold:
                return
            # OBV rising
            if i < self.obv_lookback:
                return
            obv_prev = self._obv[i - self.obv_lookback]
            if np.isnan(obv_prev) or obv_val <= obv_prev:
                return

            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. SCALE-IN
        elif side == 1 and self._should_add(i, price, atr_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, i, price, atr_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        # Strategy-specific: R² still high + slope still positive + OBV still rising
        if self._r2[i] < self.r2_threshold:
            return False
        if self._slope[i] <= 0:
            return False
        if i >= self.obv_lookback:
            obv_prev = self._obv[i - self.obv_lookback]
            if not np.isnan(obv_prev) and self._obv[i] <= obv_prev:
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
