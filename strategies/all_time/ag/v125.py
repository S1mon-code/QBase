import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.bayesian_trend import bayesian_online_trend
from indicators.regime.stationarity_score import stationarity
from indicators.volatility.ttm_squeeze import ttm_squeeze

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV125(TimeSeriesStrategy):
    """
    策略简介：贝叶斯变点检测 + 行情稳定性 + TTM Squeeze 方向的多空策略。

    使用指标：
    - Bayesian Online Trend(hazard=0.01): 检测趋势变点及方向
    - Stationarity Score(60): 行情稳定时信号更可靠
    - TTM Squeeze(20): 波动率压缩后的突破方向

    进场条件（做多）：贝叶斯趋势>0，Stationarity高（稳定行情），Squeeze momentum>0
    进场条件（做空）：贝叶斯趋势<0，Stationarity高，Squeeze momentum<0

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 贝叶斯趋势反转

    优点：变点检测对突变敏感，TTM Squeeze 捕捉爆发行情
    缺点：贝叶斯在线推理对参数敏感，可能过度响应噪音
    """
    name = "ag_alltime_v125"
    warmup = 400
    freq = "4h"

    hazard_rate: float = 0.01
    stat_period: int = 60
    stat_thresh: float = 0.5
    squeeze_bb: int = 20
    squeeze_kc: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._bayes_trend = None
        self._stationarity = None
        self._squeeze_on = None
        self._squeeze_mom = None
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

        self._atr = atr(highs, lows, closes, period=14)
        self._bayes_trend = bayesian_online_trend(closes, hazard_rate=self.hazard_rate)
        self._stationarity = stationarity(closes, period=self.stat_period)
        self._squeeze_on, self._squeeze_mom = ttm_squeeze(
            highs, lows, closes, bb=self.squeeze_bb, kc=self.squeeze_kc
        )

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
        bt_val = self._bayes_trend[i]
        stat_val = self._stationarity[i]
        sq_mom = self._squeeze_mom[i]
        if np.isnan(bt_val) or np.isnan(stat_val) or np.isnan(sq_mom):
            return

        stable = stat_val > self.stat_thresh

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
        if side == 1 and bt_val < 0:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and bt_val > 0:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and stable:
            if bt_val > 0 and sq_mom > 0:
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
            elif bt_val < 0 and sq_mom < 0:
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
            if self.direction == 1 and bt_val > 0 and sq_mom > 0:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and bt_val < 0 and sq_mom < 0:
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
