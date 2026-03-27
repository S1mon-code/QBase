import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.ml.boosting_signal import gradient_boost_signal
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV115(TimeSeriesStrategy):
    """
    策略简介：Gradient Boosting ML信号 + 传统指标特征的日线多头策略。

    使用指标：
    - Gradient Boost Signal(120, 20): 基于RSI/ADX/ROC特征的梯度提升信号
    - ATR(14): 止损

    进场：GBM信号 > 0.3
    出场：GBM信号 < -0.1 / ATR止损 / 分层止盈

    优点：GBM非线性捕捉复杂模式
    缺点：ML模型过拟合风险高
    """
    name = "medium_trend_v115"
    warmup = 120
    freq = "daily"

    gbm_period: int = 120; gbm_threshold: float = 0.3; atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._gbm = None; self._atr = None; self._avg_volume = None

    def on_init(self, context):
        self.entry_price = 0.0; self.stop_price = 0.0; self.highest_since_entry = 0.0
        self.position_scale = 0; self.bars_since_last_scale = 0
        self._took_profit_3atr = False; self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        c = context.get_full_close_array(); h = context.get_full_high_array()
        l = context.get_full_low_array(); v = context.get_full_volume_array()
        rsi_arr = rsi(c, period=14); adx_arr = adx(h, l, c, period=14)
        n = len(c); roc_arr = np.full(n, np.nan); roc_arr[10:] = (c[10:] - c[:-10]) / c[:-10]
        features = np.column_stack([rsi_arr, adx_arr, roc_arr])
        self._gbm = gradient_boost_signal(c, features, period=self.gbm_period, n_estimators=20)
        self._atr = atr(h, l, c, period=14)
        self._avg_volume = fast_avg_volume(v, 20)

    def on_bar(self, context):
        i = context.bar_index; price = context.close_raw; side, lots = context.position
        if context.is_rollover: return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1: return
        av = self._atr[i]; sig = self._gbm[i]
        if np.isnan(av) or av <= 0 or np.isnan(sig): return
        self.bars_since_last_scale += 1

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            self.stop_price = max(self.stop_price, self.highest_since_entry - self.atr_stop_mult * av)
            if price <= self.stop_price: context.close_long(); self._reset_state(); return
        if side == 1 and self.entry_price > 0:
            pa = (price - self.entry_price) / av
            if pa >= 5.0 and not self._took_profit_5atr: context.close_long(lots=max(1, lots // 3)); self._took_profit_5atr = True; return
            elif pa >= 3.0 and not self._took_profit_3atr: context.close_long(lots=max(1, lots // 3)); self._took_profit_3atr = True; return
        if side == 1 and sig < -0.1: context.close_long(); self._reset_state(); return
        if side == 0 and sig > self.gbm_threshold:
            bl = self._calc_lots(context, av)
            if bl > 0:
                context.buy(bl); self.entry_price = price; self.stop_price = price - self.atr_stop_mult * av
                self.highest_since_entry = price; self.position_scale = 1; self.bars_since_last_scale = 0
        elif side == 1 and self.position_scale < MAX_SCALE and self.bars_since_last_scale >= 10 and price > self.entry_price + av and sig > 0.2:
            f = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
            a = max(1, int(self._calc_lots(context, av) * f))
            if a > 0: context.buy(a); self.position_scale += 1; self.bars_since_last_scale = 0

    def _calc_lots(self, context, av):
        spec = _SPEC_MANAGER.get(context.symbol)
        sd = self.atr_stop_mult * av * spec.multiplier
        if sd <= 0: return 0
        rl = int(context.equity * 0.02 / sd); m = context.close_raw * spec.multiplier * spec.margin_rate
        if m <= 0: return 0
        return max(1, min(rl, int(context.equity * 0.30 / m)))

    def _reset_state(self):
        self.entry_price = 0.0; self.stop_price = 0.0; self.highest_since_entry = 0.0
        self.position_scale = 0; self.bars_since_last_scale = 0
        self._took_profit_3atr = False; self._took_profit_5atr = False
