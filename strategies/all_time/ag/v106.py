import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.hma import hma
from indicators.structure.position_crowding import position_crowding
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class HMACrowdingStrategy(TimeSeriesStrategy):
    """
    策略简介：HMA趋势方向 + Position Crowding拥挤度管理的多空策略。

    使用指标：
    - HMA(20): Hull均线方向及斜率判断趋势
    - Position Crowding(60): 持仓拥挤度评分，极端拥挤时减仓/不加仓
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - HMA斜率 > 0（上升趋势）
    - Crowding score < 0.7（持仓不过度拥挤）

    进场条件（做空）：
    - HMA斜率 < 0（下降趋势）
    - Crowding score < 0.7（持仓不过度拥挤）

    出场条件：
    - ATR追踪止损 / 分层止盈 / HMA方向反转 / Crowding极端（unwind_risk > 0.8）

    优点：HMA响应快延迟低，crowding过滤高反转风险区间
    缺点：HMA在震荡市频繁翻转，crowding需要OI数据
    """
    name = "v106_hma_crowding"
    warmup = 300
    freq = "4h"

    hma_period: int = 20            # Optuna: 10-40
    crowding_period: int = 60       # Optuna: 30-120
    crowding_threshold: float = 0.7 # Optuna: 0.5-0.9
    atr_stop_mult: float = 3.0     # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._hma = None
        self._crowding = None
        self._unwind_risk = None
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
        oi = context.get_full_oi_array()

        self._hma = hma(closes, period=self.hma_period)
        self._crowding, self._unwind_risk = position_crowding(
            closes, oi, volumes, period=self.crowding_period)
        self._atr = atr(highs, lows, closes, period=14)

        window = 20
        self._avg_volume = np.full_like(volumes, np.nan)
        for idx in range(window, len(volumes)):
            self._avg_volume[idx] = np.mean(volumes[idx - window:idx])

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        hma_val = self._hma[i]
        hma_prev = self._hma[i - 1] if i > 0 else np.nan
        if np.isnan(hma_val) or np.isnan(hma_prev):
            return
        hma_slope = hma_val - hma_prev
        crowd = self._crowding[i] if not np.isnan(self._crowding[i]) else 0.0
        unwind = self._unwind_risk[i] if not np.isnan(self._unwind_risk[i]) else 0.0

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
            if side == 1:
                profit_atr = (price - self.entry_price) / atr_val
            else:
                profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_3atr = True
                return

        # 3. Signal-based exit: HMA flip or extreme crowding
        if side == 1 and (hma_slope < 0 or unwind > 0.8):
            context.close_long()
            self._reset_state()
        elif side == -1 and (hma_slope > 0 or unwind > 0.8):
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry
        if side == 0:
            if hma_slope > 0 and crowd < self.crowding_threshold:
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
            elif hma_slope < 0 and crowd < self.crowding_threshold:
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

        # 5. Scale-in (blocked when crowded)
        elif side != 0 and self._should_add(price, atr_val) and crowd < self.crowding_threshold:
            if self.direction == 1 and hma_slope > 0:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and hma_slope < 0:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
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
        self.direction = 0
