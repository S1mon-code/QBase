import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.regime.changepoint import changepoint_score
from indicators.regime.trend_persistence import trend_persistence
from indicators.volume.obv import obv

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV147(TimeSeriesStrategy):
    """
    策略简介：贝叶斯变点检测 + 趋势稳定性 + OBV背离的多空策略。

    使用指标：
    - Changepoint Score(60): 变点概率检测，高概率=regime即将切换
    - Trend Persistence(60): 趋势稳定性，高值=稳定趋势
    - OBV: 量价背离信号，OBV斜率与价格斜率不一致
    - ATR(14): 止损距离计算

    进场条件（做多）：变点触发(>0.7) + 稳定regime(persistence>0.5) + OBV向上
    进场条件（做空）：变点触发(>0.7) + 稳定regime(persistence>0.5) + OBV向下

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 新变点出现且方向不利

    优点：变点检测捕捉regime切换时机，OBV确认量价一致
    缺点：变点检测有滞后，日频信号稀少
    """
    name = "ag_alltime_v147"
    warmup = 250
    freq = "daily"

    cp_threshold: float = 0.7       # Optuna: 0.5-0.9
    persist_threshold: float = 0.5  # Optuna: 0.3-0.7
    obv_lookback: int = 20          # Optuna: 10-40
    atr_stop_mult: float = 3.5     # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._avg_volume = None
        self._cp_score = None
        self._persist = None
        self._obv = None

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

        self._atr = atr(highs, lows, closes, period=14)
        self._cp_score = changepoint_score(closes, period=60)
        self._persist, _ = trend_persistence(closes, max_lag=20, period=60)
        self._obv = obv(closes, volumes)

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
        cp_val = self._cp_score[i]
        persist_val = self._persist[i]
        if np.isnan(cp_val) or np.isnan(persist_val):
            return
        if i < self.obv_lookback:
            return

        obv_now = self._obv[i]
        obv_prev = self._obv[i - self.obv_lookback]
        if np.isnan(obv_now) or np.isnan(obv_prev):
            return
        obv_slope = obv_now - obv_prev

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

        # 3. Signal exit: new changepoint against position
        if side == 1 and cp_val > self.cp_threshold and obv_slope < 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and cp_val > self.cp_threshold and obv_slope > 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry
        if side == 0 and cp_val > self.cp_threshold and persist_val > self.persist_threshold:
            if obv_slope > 0:
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
            elif obv_slope < 0:
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
        elif side != 0 and self._should_add(price, atr_val, obv_slope):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if self.direction == 1:
                    context.buy(add_lots)
                else:
                    context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, obv_slope):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1:
            if price < self.entry_price + atr_val:
                return False
            if obv_slope <= 0:
                return False
        elif self.direction == -1:
            if price > self.entry_price - atr_val:
                return False
            if obv_slope >= 0:
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
