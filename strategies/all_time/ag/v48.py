import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.decision_boundary import decision_boundary_distance
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class SVMSupertrendStrategy(TimeSeriesStrategy):
    """
    策略简介：SVM决策边界距离（方向置信度）+ Supertrend趋势过滤。

    使用指标：
    - Decision Boundary Distance (SVM proxy): 当前bar到分类边界的距离，
      正=看多侧，负=看空侧，绝对值越大越确信
    - Supertrend(10, 3.0): 趋势方向过滤，1=上升，-1=下降
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - SVM距离 > threshold（分类为看多且置信度高）
    - Supertrend方向 = 1（趋势向上）

    进场条件（做空）：
    - SVM距离 < -threshold（分类为看空且置信度高）
    - Supertrend方向 = -1（趋势向下）

    出场条件：
    - ATR追踪止损 / 分层止盈 / Supertrend反转

    优点：SVM决策边界给出清晰的方向和置信度，Supertrend简单可靠
    缺点：线性SVM在非线性市场中可能不够灵活
    """
    name = "v48_svm_supertrend"
    warmup = 500
    freq = "4h"

    svm_period: int = 120
    svm_threshold: float = 0.5
    st_period: int = 10
    st_mult: float = 3.0
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._svm_dist = None
        self._st_dir = None
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

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        n = len(closes)

        rsi_arr = rsi(closes, 14)
        atr_arr = atr(highs, lows, closes, 14)
        returns = np.full(n, np.nan)
        returns[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
        features = np.column_stack([rsi_arr, atr_arr / np.maximum(closes, 1e-9), returns])

        # Create binary labels for SVM: 1 if next-bar return > 0, else 0
        labels = np.full(n, np.nan)
        for idx in range(n - 1):
            if not np.isnan(returns[idx + 1]):
                labels[idx] = 1.0 if returns[idx + 1] > 0 else 0.0

        self._svm_dist = decision_boundary_distance(
            features, labels, period=self.svm_period)

        _, self._st_dir = supertrend(highs, lows, closes, self.st_period, self.st_mult)
        self._atr = atr(highs, lows, closes, period=self.atr_period)

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        svm_val = self._svm_dist[i]
        st_dir = self._st_dir[i]
        atr_val = self._atr[i]
        if np.isnan(svm_val) or np.isnan(st_dir) or np.isnan(atr_val) or atr_val <= 0:
            return

        self.bars_since_last_scale += 1

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

        # ── 3. Supertrend反转退出 ──
        if side == 1 and st_dir == -1:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and st_dir == 1:
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场 ──
        if side == 0:
            if svm_val > self.svm_threshold and st_dir == 1:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
            elif svm_val < -self.svm_threshold and st_dir == -1:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0

        # ── 5. 加仓 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and svm_val > self.svm_threshold and st_dir == 1):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and svm_val < -self.svm_threshold and st_dir == -1):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.sell(add)
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
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
