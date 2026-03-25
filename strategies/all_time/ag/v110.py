import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.volatility.hurst import hurst_exponent
from indicators.spread.gold_silver_ratio import gold_silver_ratio
from indicators.spread.ratio_momentum import ratio_momentum

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV110(TimeSeriesStrategy):
    """
    策略简介：Hurst指数判断市场状态 + 金银比动量的自适应策略。

    使用指标：
    - Hurst Exponent(20): H>0.5趋势，H<0.5均值回归
    - Gold/Silver Ratio(60): 金银比及其z-score
    - Ratio Momentum(20): 金银比率的变化速度
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - 趋势模式(H>0.55): 金银比下降（银相对强势），ratio_roc<0
    - 均值回归模式(H<0.45): 金银比z-score>1.5（银超跌反弹）
    进场条件（做空）：
    - 趋势模式(H>0.55): 金银比上升（银相对弱势），ratio_roc>0
    - 均值回归模式(H<0.45): 金银比z-score<-1.5（银超涨回落）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Hurst状态改变或比率信号反转

    优点：自适应切换趋势/均值回归模式，利用金银比的经济学关系
    缺点：Hurst计算有滞后，金银比受宏观事件冲击可能失效
    """
    name = "ag_alltime_v110"
    warmup = 300
    freq = "daily"

    hurst_lag: int = 20           # Optuna: 15-30
    hurst_trend: float = 0.55     # Optuna: 0.52-0.60
    hurst_mr: float = 0.45        # Optuna: 0.40-0.48
    gsr_period: int = 60          # Optuna: 40-120
    ratio_roc_period: int = 20    # Optuna: 10-30
    mr_zscore_thresh: float = 1.5 # Optuna: 1.0-2.5
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._hurst = None
        self._gs_ratio = None
        self._gs_zscore = None
        self._ratio_roc = None
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
        n = len(closes)

        self._atr = atr(highs, lows, closes, period=14)
        self._hurst = hurst_exponent(closes, max_lag=self.hurst_lag)

        # Load AU auxiliary close for gold/silver ratio
        au_closes = context.load_auxiliary_close("AU")
        if au_closes is not None and len(au_closes) == n:
            self._gs_ratio, self._gs_zscore = gold_silver_ratio(
                au_closes, closes, period=self.gsr_period
            )
            self._ratio_roc, _ = ratio_momentum(
                au_closes, closes, period=self.ratio_roc_period
            )
        else:
            self._gs_ratio = np.full(n, np.nan)
            self._gs_zscore = np.full(n, np.nan)
            self._ratio_roc = np.full(n, np.nan)

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
        h_val = self._hurst[i]
        gs_z = self._gs_zscore[i]
        roc_val = self._ratio_roc[i]
        if np.isnan(h_val) or np.isnan(gs_z) or np.isnan(roc_val):
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

        # Determine regime
        is_trending = h_val > self.hurst_trend
        is_mr = h_val < self.hurst_mr

        # 3. Signal-based exit
        if side == 1:
            if is_trending and roc_val > 0:
                # Ratio rising = silver weakening in trend mode
                context.close_long()
                self._reset_state()
                return
            if is_mr and gs_z < 0:
                # Mean reversion: z-score flipped
                context.close_long()
                self._reset_state()
                return
        if side == -1:
            if is_trending and roc_val < 0:
                context.close_short()
                self._reset_state()
                return
            if is_mr and gs_z > 0:
                context.close_short()
                self._reset_state()
                return

        # Re-read position
        side, lots = context.position

        # 4. Entry logic
        if side == 0:
            # Trending mode: follow ratio momentum
            if is_trending:
                if roc_val < 0:
                    # Ratio falling = silver strengthening -> long silver
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
                elif roc_val > 0:
                    # Ratio rising = silver weakening -> short silver
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
            # Mean reversion mode: fade extreme z-scores
            elif is_mr:
                if gs_z > self.mr_zscore_thresh:
                    # Silver undervalued -> long
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
                elif gs_z < -self.mr_zscore_thresh:
                    # Silver overvalued -> short
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
            signal_ok = False
            if is_trending:
                signal_ok = (self.direction == 1 and roc_val < 0) or \
                            (self.direction == -1 and roc_val > 0)
            elif is_mr:
                signal_ok = (self.direction == 1 and gs_z > 0.5) or \
                            (self.direction == -1 and gs_z < -0.5)
            if signal_ok:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    if self.direction == 1:
                        context.buy(add_lots)
                    else:
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
