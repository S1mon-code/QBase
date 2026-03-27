import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.momentum.roc import rate_of_change
from indicators.trend.ema_ribbon import ema_ribbon
from indicators.structure.position_crowding import position_crowding
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV192(TimeSeriesStrategy):
    """
    策略简介：ROC动量+EMA Ribbon趋势扩张+持仓拥挤度仓位管理的趋势策略。

    使用指标：
    - ROC(12): 价格变化率，方向判断
    - EMA Ribbon(8,13,21,34,55): 多均线排列，扩张=趋势强
    - Position Crowding(60): 拥挤度评分，拥挤时缩减仓位
    - ATR(14): 止损距离计算

    进场条件（做多）：ROC>0 + EMA Ribbon完全多头排列（短>长）+ 拥挤度低
    进场条件（做空）：ROC<0 + EMA Ribbon完全空头排列（长>短）+ 拥挤度低

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - EMA Ribbon排列破坏

    优点：Ribbon提供清晰的趋势结构，拥挤度管理避免拥挤反转风险
    缺点：完全排列条件严格，入场机会较少
    """
    name = "ag_alltime_v192"
    warmup = 300
    freq = "4h"

    roc_period: int = 12          # Optuna: 8-20
    crowding_period: int = 60     # Optuna: 40-80
    crowding_thresh: float = 0.7  # Optuna: 0.5-0.9
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._roc = None
        self._ribbon = None
        self._crowding = None
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

        self._roc = rate_of_change(closes, self.roc_period)
        self._ribbon = ema_ribbon(closes, periods=(8, 13, 21, 34, 55))
        self._crowding, _ = position_crowding(closes, oi, volumes, self.crowding_period)
        self._atr = atr(highs, lows, closes, period=14)

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def _ribbon_bullish(self, i):
        """Check if EMA ribbon is in full bullish alignment (short > long)."""
        for k in range(len(self._ribbon) - 1):
            v1 = self._ribbon[k][i]
            v2 = self._ribbon[k + 1][i]
            if np.isnan(v1) or np.isnan(v2) or v1 <= v2:
                return False
        return True

    def _ribbon_bearish(self, i):
        """Check if EMA ribbon is in full bearish alignment (long > short)."""
        for k in range(len(self._ribbon) - 1):
            v1 = self._ribbon[k][i]
            v2 = self._ribbon[k + 1][i]
            if np.isnan(v1) or np.isnan(v2) or v1 >= v2:
                return False
        return True

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        roc_val = self._roc[i]
        crowd = self._crowding[i]

        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(roc_val):
            return
        if np.isnan(crowd):
            crowd = 0.0

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

        # 3. Signal exit: ribbon alignment breaks
        bull = self._ribbon_bullish(i)
        bear = self._ribbon_bearish(i)
        if side == 1 and not bull:
            context.close_long()
            self._reset_state()
        elif side == -1 and not bear:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: ROC direction + ribbon alignment + low crowding
        not_crowded = crowd < self.crowding_thresh
        if side == 0:
            if roc_val > 0 and self._ribbon_bullish(i) and not_crowded:
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
            elif roc_val < 0 and self._ribbon_bearish(i) and not_crowded:
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

        # 5. Scale-in (skip if crowded)
        elif side != 0 and self._should_add(price, atr_val) and not_crowded:
            if self.direction == 1 and roc_val > 0 and self._ribbon_bullish(i):
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and roc_val < 0 and self._ribbon_bearish(i):
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
