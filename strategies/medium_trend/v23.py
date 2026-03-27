import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.boosting_signal import gradient_boost_signal
from indicators.regime.trend_strength_composite import trend_strength
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV23(TimeSeriesStrategy):
    """
    策略简介：Gradient Boosting ML信号 + Trend Strength过滤的做多策略（10min频率）。

    使用指标：
    - Gradient Boost Signal(120,20): ML集成学习趋势预测
    - Trend Strength Composite(20): 趋势状态确认
    - ATR(14): 止损距离计算

    进场条件（做多）：boosting signal > 0 且 trend_strength > 0.5
    出场条件：ATR追踪止损 / 分层止盈 / boosting signal < 0

    优点：ML捕捉非线性模式，trend_strength过滤虚假信号
    缺点：ML模型可能过拟合历史特征
    """
    name = "mt_v23"
    warmup = 1200
    freq = "10min"

    boost_period: int = 120
    boost_n_est: int = 20
    ts_period: int = 20
    ts_threshold: float = 0.5
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._boost = None
        self._ts = None
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

        rsi_arr = rsi(closes, 14)
        adx_arr = adx(highs, lows, closes, 14)
        atr_arr = atr(highs, lows, closes, 14)
        features = np.column_stack([rsi_arr, adx_arr, atr_arr])

        self._boost = gradient_boost_signal(closes, features, period=self.boost_period,
                                             n_estimators=self.boost_n_est)
        self._ts = trend_strength(closes, highs, lows, period=self.ts_period)
        self._atr = atr_arr
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
        boost = self._boost[i]
        ts_val = self._ts[i]
        if np.isnan(atr_val) or np.isnan(boost) or np.isnan(ts_val):
            return

        self.bars_since_last_scale += 1
        bullish = boost > 0 and ts_val > self.ts_threshold

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

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

        if side == 1 and boost < 0:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and bullish:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, bullish):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, bullish):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if not bullish:
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
