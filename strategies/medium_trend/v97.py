import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.bayesian_trend import bayesian_online_trend
from indicators.volume.klinger import klinger
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV97(TimeSeriesStrategy):
    """
    策略简介：Bayesian Online Trend变点检测 + Klinger Oscillator量能确认的4h多头策略。

    使用指标：
    - Bayesian Online Trend(0.01): 检测趋势变化点，>0.5为上升概率
    - Klinger Oscillator(34,55,13): 量能震荡器确认资金方向
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Bayesian趋势概率 > 0.6（高概率上升趋势）
    - Klinger > 信号线（量能多头）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - Bayesian趋势概率 < 0.3（下降概率增大）

    优点：贝叶斯方法自适应，变点检测快
    缺点：Hazard Rate参数敏感，计算量大
    """
    name = "medium_trend_v97"
    warmup = 300
    freq = "4h"

    hazard_rate: float = 0.01     # Optuna: 0.005-0.05
    trend_threshold: float = 0.6  # Optuna: 0.5-0.8
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._bay_trend = None
        self._kvo = None
        self._kvo_signal = None
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

        self._bay_trend = bayesian_online_trend(closes, hazard_rate=self.hazard_rate)
        self._kvo, self._kvo_signal = klinger(highs, lows, closes, volumes)
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

        atr_val = self._atr[i]
        bay = self._bay_trend[i]
        kvo = self._kvo[i]
        kvo_sig = self._kvo_signal[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(bay) or np.isnan(kvo) or np.isnan(kvo_sig):
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

        # 3. Signal exit: trend probability low
        if side == 1 and bay < 0.3:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and bay > self.trend_threshold and kvo > kvo_sig:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, bay, kvo, kvo_sig):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, bay, kvo, kvo_sig):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if bay < self.trend_threshold or kvo <= kvo_sig:
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
