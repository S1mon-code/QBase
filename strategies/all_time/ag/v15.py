import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.momentum.roc import rate_of_change

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _seasonal_momentum(closes, datetimes, lookback_years=3):
    """Seasonal momentum: average return for same calendar month over past years.

    Returns (seasonal_score, seasonal_strength) arrays.
    seasonal_score > 0 = historically bullish month
    seasonal_score < 0 = historically bearish month
    seasonal_strength = consistency (0-1)
    """
    n = len(closes)
    score = np.full(n, np.nan, dtype=np.float64)
    strength = np.full(n, np.nan, dtype=np.float64)

    if n < 252 * lookback_years:
        return score, strength

    # Extract months
    months = np.array([dt.month for dt in datetimes])
    years = np.array([dt.year for dt in datetimes])

    # Monthly returns: approximate by looking at same month in prior years
    for i in range(252, n):
        current_month = months[i]
        current_year = years[i]

        hist_returns = []
        for y_offset in range(1, lookback_years + 1):
            target_year = current_year - y_offset
            # Find bars in same month of that year
            mask = (months == current_month) & (years == target_year)
            indices = np.where(mask)[0]
            if len(indices) < 2:
                continue
            first_idx = indices[0]
            last_idx = indices[-1]
            if closes[first_idx] > 0:
                ret = (closes[last_idx] / closes[first_idx] - 1.0) * 100
                hist_returns.append(ret)

        if len(hist_returns) >= 2:
            score[i] = np.mean(hist_returns)
            # Consistency: fraction of same-sign returns
            signs = np.sign(hist_returns)
            dominant = max(np.sum(signs > 0), np.sum(signs < 0))
            strength[i] = dominant / len(hist_returns)

    return score, strength


class StrategyV15(TimeSeriesStrategy):
    """
    策略简介：基于季节性动量模式的交易策略。

    使用指标：
    - Seasonal Momentum(3yr): 同月历史收益率均值
    - ROC(20): 当前动量确认
    - ATR(14): 止损距离计算

    进场条件（做多）：季节性得分>1.0, 一致性>0.6, ROC>0
    进场条件（做空）：季节性得分<-1.0, 一致性>0.6, ROC<0

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 月度切换时季节性信号消失

    优点：利用白银已知的季节性规律（如年末需求旺季）
    缺点：季节性可能被基本面变化打破，样本量有限
    """
    name = "ag_alltime_v15"
    warmup = 800  # Need 3+ years lookback
    freq = "daily"

    lookback_years: int = 3       # Optuna: 2-5
    score_thresh: float = 1.0    # Optuna: 0.5-2.0
    consistency_thresh: float = 0.6  # Optuna: 0.5-0.8
    roc_period: int = 20          # Optuna: 10-30
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._seasonal_score = None
        self._seasonal_str = None
        self._roc = None
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
        datetimes = context.get_full_datetime_array()

        self._atr = atr(highs, lows, closes, period=14)
        self._roc = rate_of_change(closes, period=self.roc_period)
        self._seasonal_score, self._seasonal_str = _seasonal_momentum(
            closes, datetimes, lookback_years=self.lookback_years
        )

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

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
        ss = self._seasonal_score[i]
        sstr = self._seasonal_str[i]
        roc_val = self._roc[i]
        if np.isnan(ss) or np.isnan(sstr) or np.isnan(roc_val):
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

        # 3. Signal-based exit: momentum reversal
        if side == 1 and roc_val < 0 and ss < 0:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and roc_val > 0 and ss > 0:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if ss > self.score_thresh and sstr > self.consistency_thresh and roc_val > 0:
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
            elif ss < -self.score_thresh and sstr > self.consistency_thresh and roc_val < 0:
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
            if self.direction == 1 and ss > 0 and roc_val > 0:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and ss < 0 and roc_val < 0:
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
