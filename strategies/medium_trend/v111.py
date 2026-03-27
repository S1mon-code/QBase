import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.linear_regression import linear_regression_slope
from indicators.momentum.stochastic import stochastic
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV111(TimeSeriesStrategy):
    """
    策略简介：Linear Regression Slope方向 + Stochastic入场的日线多头策略。

    使用指标：
    - LR Slope(20): > 0确认价格线性上升趋势
    - Stochastic(14,3): %K从超卖回升精准入场
    - ATR(14): 止损

    进场：LR Slope > 0 + Stoch %K从<20升至>20
    出场：LR Slope < 0 / Stoch > 80 / ATR止损 / 分层止盈

    优点：线性回归斜率稳定，Stochastic精准择时
    缺点：线性假设在非线性行情中失效
    """
    name = "medium_trend_v111"
    warmup = 80
    freq = "daily"

    lr_period: int = 20; stoch_k: int = 14; stoch_d: int = 3
    stoch_oversold: float = 20.0; atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._slope = None; self._sk = None; self._atr = None; self._avg_volume = None

    def on_init(self, context):
        self.entry_price = 0.0; self.stop_price = 0.0; self.highest_since_entry = 0.0
        self.position_scale = 0; self.bars_since_last_scale = 0
        self._took_profit_3atr = False; self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        c = context.get_full_close_array(); h = context.get_full_high_array()
        l = context.get_full_low_array(); v = context.get_full_volume_array()
        self._slope = linear_regression_slope(c, self.lr_period)
        self._sk, _ = stochastic(h, l, c, k=self.stoch_k, d=self.stoch_d)
        self._atr = atr(h, l, c, period=14)
        self._avg_volume = fast_avg_volume(v, 20)

    def on_bar(self, context):
        i = context.bar_index; price = context.close_raw; side, lots = context.position
        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover: return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1: return
        av = self._atr[i]; sl = self._slope[i]; sk = self._sk[i]
        if np.isnan(av) or av <= 0 or np.isnan(sl) or np.isnan(sk): return
        sk_prev = self._sk[i - 1] if i > 0 else np.nan
        if np.isnan(sk_prev): return
        self.bars_since_last_scale += 1

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            self.stop_price = max(self.stop_price, self.highest_since_entry - self.atr_stop_mult * av)
            if price <= self.stop_price: context.close_long(); self._reset_state(); return
        if side == 1 and self.entry_price > 0:
            pa = (price - self.entry_price) / av
            if pa >= 5.0 and not self._took_profit_5atr: context.close_long(lots=max(1, lots // 3)); self._took_profit_5atr = True; return
            elif pa >= 3.0 and not self._took_profit_3atr: context.close_long(lots=max(1, lots // 3)); self._took_profit_3atr = True; return
        if side == 1 and (sl < 0 or sk > 80): context.close_long(); self._reset_state(); return
        if side == 0 and sl > 0 and sk_prev < self.stoch_oversold and sk >= self.stoch_oversold:
            bl = self._calc_lots(context, av)
            if bl > 0:
                context.buy(bl); self.entry_price = price; self.stop_price = price - self.atr_stop_mult * av
                self.highest_since_entry = price; self.position_scale = 1; self.bars_since_last_scale = 0
        elif side == 1 and self.position_scale < MAX_SCALE and self.bars_since_last_scale >= 10 and price > self.entry_price + av and sl > 0:
            f = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
            a = max(1, int(self._calc_lots(context, av) * f))
            if a > 0: context.buy(a); self.position_scale += 1; self.bars_since_last_scale = 0

    def _calc_lots(self, context, av):
        from alphaforge.data.contract_specs import ContractSpecManager
        spec = ContractSpecManager().get(context.symbol)
        sd = self.atr_stop_mult * av * spec.multiplier
        if sd <= 0: return 0
        rl = int(context.equity * 0.02 / sd); m = context.close_raw * spec.multiplier * spec.margin_rate
        if m <= 0: return 0
        return max(1, min(rl, int(context.equity * 0.30 / m)))

    def _reset_state(self):
        self.entry_price = 0.0; self.stop_price = 0.0; self.highest_since_entry = 0.0
        self.position_scale = 0; self.bars_since_last_scale = 0
        self._took_profit_3atr = False; self._took_profit_5atr = False
