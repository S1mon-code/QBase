import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.momentum.macd import macd
from indicators.volume.volume_price_trend import vpt
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV107(TimeSeriesStrategy):
    """
    策略简介：MACD趋势动量 + VPT量价趋势确认的日线多头策略。

    使用指标：
    - MACD(12,26,9): 柱状图>0且线>信号线确认上涨动量
    - VPT: Volume Price Trend上升确认资金持续流入
    - ATR(14): 止损距离

    进场条件（做多）：MACD hist > 0 + VPT上升
    出场条件：MACD hist < 0 / ATR止损 / 分层止盈

    优点：MACD+VPT跨价格和量双维度确认
    缺点：MACD滞后，VPT累积值可能长期偏向一方
    """
    name = "medium_trend_v107"
    warmup = 80
    freq = "daily"

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    vpt_lookback: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._macd_hist = None
        self._vpt = None
        self._atr = None
        self._avg_volume = None

    def on_init(self, context):
        self.entry_price = 0.0; self.stop_price = 0.0; self.highest_since_entry = 0.0
        self.position_scale = 0; self.bars_since_last_scale = 0
        self._took_profit_3atr = False; self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        _, _, self._macd_hist = macd(closes, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        self._vpt = vpt(closes, volumes)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover: return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1: return
        atr_val = self._atr[i]; hist = self._macd_hist[i]; vpt_val = self._vpt[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(hist) or np.isnan(vpt_val): return
        vpt_prev = self._vpt[i - self.vpt_lookback] if i >= self.vpt_lookback else np.nan
        vpt_rising = not np.isnan(vpt_prev) and vpt_val > vpt_prev
        self.bars_since_last_scale += 1

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            self.stop_price = max(self.stop_price, self.highest_since_entry - self.atr_stop_mult * atr_val)
            if price <= self.stop_price: context.close_long(); self._reset_state(); return
        if side == 1 and self.entry_price > 0:
            pa = (price - self.entry_price) / atr_val
            if pa >= 5.0 and not self._took_profit_5atr: context.close_long(lots=max(1, lots // 3)); self._took_profit_5atr = True; return
            elif pa >= 3.0 and not self._took_profit_3atr: context.close_long(lots=max(1, lots // 3)); self._took_profit_3atr = True; return
        if side == 1 and hist < 0: context.close_long(); self._reset_state(); return
        if side == 0 and hist > 0 and vpt_rising:
            bl = self._calc_lots(context, atr_val)
            if bl > 0:
                context.buy(bl); self.entry_price = price; self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price; self.position_scale = 1; self.bars_since_last_scale = 0
        elif side == 1 and self.position_scale < MAX_SCALE and self.bars_since_last_scale >= 10 and price > self.entry_price + atr_val and hist > 0:
            f = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
            a = max(1, int(self._calc_lots(context, atr_val) * f))
            if a > 0: context.buy(a); self.position_scale += 1; self.bars_since_last_scale = 0

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        sd = self.atr_stop_mult * atr_val * spec.multiplier
        if sd <= 0: return 0
        rl = int(context.equity * 0.02 / sd)
        m = context.close_raw * spec.multiplier * spec.margin_rate
        if m <= 0: return 0
        return max(1, min(rl, int(context.equity * 0.30 / m)))

    def _reset_state(self):
        self.entry_price = 0.0; self.stop_price = 0.0; self.highest_since_entry = 0.0
        self.position_scale = 0; self.bars_since_last_scale = 0
        self._took_profit_3atr = False; self._took_profit_5atr = False
