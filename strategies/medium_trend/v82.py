import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.adx import adx
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV82(TimeSeriesStrategy):
    """
    策略简介：ADX趋势强度 + ROC动量确认 + ATR波动率过滤的4h多头趋势策略。

    使用指标：
    - ADX(14): 趋势强度 > threshold确认趋势存在
    - ROC(20): 动量方向确认，> 0时确认上涨
    - ATR(14): 止损距离计算 + 波动率过滤

    进场条件（做多）：
    - ADX > adx_threshold（趋势够强）
    - ROC > 0（动量向上）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - ROC < 0（动量反转）

    优点：双重确认（趋势+动量），减少假信号
    缺点：ADX滞后，震荡市入场偏晚
    """
    name = "medium_trend_v82"
    warmup = 200
    freq = "4h"

    adx_period: int = 14         # Optuna: 10-20
    roc_period: int = 20         # Optuna: 10-30
    adx_threshold: float = 25.0  # Optuna: 15-40
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._adx = None
        self._roc = None
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

        self._adx = adx(highs, lows, closes, period=self.adx_period)
        self._roc = rate_of_change(closes, self.roc_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        adx_val = self._adx[i]
        roc_val = self._roc[i]
        atr_val = self._atr[i]
        if np.isnan(adx_val) or np.isnan(roc_val) or np.isnan(atr_val) or atr_val <= 0:
            return

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

        # 2. Tiered profit-taking
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

        # 3. Signal exit: momentum reversal
        if side == 1 and roc_val < 0:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and adx_val > self.adx_threshold and roc_val > 0:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, adx_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, adx_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if adx_val < self.adx_threshold:
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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
