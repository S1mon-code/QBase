import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.regime.composite_regime_score import composite_regime
from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV100(TimeSeriesStrategy):
    """
    策略简介：Composite Regime Score状态识别 + Schaff Trend Cycle入场的4h多头策略。

    使用指标：
    - Composite Regime Score(20): 综合趋势/波动/量能评分识别行情状态
    - Schaff Trend Cycle(10, 23, 50): STC > 25从超卖回升入场
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Composite Regime > 0（综合评分看多）
    - STC从 < 25回升至 > 25（趋势型超卖回升）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - STC > 75后回落至 < 75（超买区弱化）

    优点：Regime Score综合多维度，STC入场精准
    缺点：Composite指标内部可能互相矛盾
    """
    name = "medium_trend_v100"
    warmup = 300
    freq = "4h"

    regime_period: int = 20        # Optuna: 14-30
    stc_period: int = 10           # Optuna: 8-15
    stc_fast: int = 23
    stc_slow: int = 50
    stc_entry: float = 25.0        # Optuna: 20-35
    atr_stop_mult: float = 3.0     # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._regime = None
        self._stc = None
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

        self._regime = composite_regime(closes, highs, lows, period=self.regime_period)
        self._stc = schaff_trend_cycle(closes, period=self.stc_period, fast=self.stc_fast, slow=self.stc_slow)
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
        regime_val = self._regime[i]
        stc_val = self._stc[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(regime_val) or np.isnan(stc_val):
            return
        stc_prev = self._stc[i - 1] if i > 0 else np.nan
        if np.isnan(stc_prev):
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

        # 3. Signal exit: STC overbought pullback
        if side == 1 and stc_prev > 75 and stc_val < 75:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry: bullish regime + STC oversold bounce
        stc_bounce = stc_prev < self.stc_entry and stc_val >= self.stc_entry
        if side == 0 and regime_val > 0 and stc_bounce:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, regime_val, stc_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, regime_val, stc_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if regime_val <= 0 or stc_val > 70:
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
