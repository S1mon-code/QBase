import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.decision_boundary import decision_boundary_distance
from indicators.regime.market_state import market_state
from indicators.spread.gold_silver_ratio import gold_silver_ratio

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV123(TimeSeriesStrategy):
    """
    策略简介：SVM-like 决策边界距离 + 市场阶段识别 + 金银比的多空策略。

    使用指标：
    - Decision Boundary Distance(120): SVM 风格分类距离作为信号强度
    - Market State(20): 识别accumulation/distribution阶段
    - Gold/Silver Ratio(60): 贵金属比值均值回归信号

    进场条件（做多）：Market State 看多，边界距离>0（分类为多），金银比下降
    进场条件（做空）：Market State 看空，边界距离<0（分类为空），金银比上升

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 边界距离反转

    优点：金银比提供基本面视角，SVM 分类信号明确
    缺点：金银比长期趋势可能掩盖短期信号
    """
    name = "ag_alltime_v123"
    warmup = 400
    freq = "daily"

    svm_period: int = 120
    ms_period: int = 20
    gsr_period: int = 60
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._boundary = None
        self._market_st = None
        self._gsr = None
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

        self._atr = atr(highs, lows, closes, period=14)

        # SVM-like: build features and labels
        from indicators.momentum.rsi import rsi
        from indicators.trend.adx import adx as adx_fn
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx_fn(highs, lows, closes, period=14)
        features = np.column_stack([rsi_arr, adx_arr, self._atr])
        # Labels: future return sign (shifted by 1 to avoid lookahead)
        n = len(closes)
        labels = np.zeros(n, dtype=np.int32)
        for idx in range(1, n):
            labels[idx - 1] = 1 if closes[idx] > closes[idx - 1] else 0
        labels[-1] = 0
        self._boundary = decision_boundary_distance(features, labels, period=self.svm_period)

        self._market_st = market_state(closes, volumes, oi, period=self.ms_period)

        # Gold/Silver ratio
        au_closes = context.load_auxiliary_close("AU")
        self._gsr = gold_silver_ratio(au_closes, closes, period=self.gsr_period)

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
        bd_val = self._boundary[i]
        ms_val = self._market_st[i]
        gsr_val = self._gsr[i]
        if np.isnan(bd_val) or np.isnan(ms_val) or np.isnan(gsr_val):
            return

        gsr_prev = self._gsr[i - 1] if i > 0 else np.nan
        if np.isnan(gsr_prev):
            return
        gsr_falling = gsr_val < gsr_prev
        gsr_rising = gsr_val > gsr_prev

        # Market state: 1=bullish accumulation, -1=bearish distribution
        ms_bull = ms_val > 0
        ms_bear = ms_val < 0

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

        # 3. Signal-based exit
        if side == 1 and bd_val < 0:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and bd_val > 0:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if bd_val > 0 and ms_bull and gsr_falling:
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
            elif bd_val < 0 and ms_bear and gsr_rising:
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

        # 5. Scale-in
        elif side != 0 and self._should_add(price, atr_val):
            if self.direction == 1 and bd_val > 0:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and bd_val < 0:
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
