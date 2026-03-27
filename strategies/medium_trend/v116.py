import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.trend.vidya import vidya
from indicators.volatility.bollinger import bollinger_bands
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV116(TimeSeriesStrategy):
    """
    策略简介：VIDYA自适应均线 + Bollinger Band宽度过滤的日线多头策略。

    使用指标：
    - VIDYA(14,9): 波动率自适应移动平均，上升为多头
    - Bollinger Bands(20,2): BB宽度作为波动率过滤
    - ATR(14): 止损

    进场：价格>VIDYA + VIDYA上升 + BB宽度适中
    出场：价格<VIDYA / ATR止损 / 分层止盈

    优点：VIDYA在趋势中自动加速跟踪
    缺点：波动率骤变时适应需要时间
    """
    name = "medium_trend_v116"
    warmup = 80
    freq = "daily"

    vidya_period: int = 14; vidya_cmo: int = 9; atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._vidya = None; self._bb_u = None; self._bb_m = None; self._bb_l = None
        self._atr = None; self._avg_volume = None

    def on_init(self, context):
        self.entry_price = 0.0; self.stop_price = 0.0; self.highest_since_entry = 0.0
        self.position_scale = 0; self.bars_since_last_scale = 0
        self._took_profit_3atr = False; self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        c = context.get_full_close_array(); h = context.get_full_high_array()
        l = context.get_full_low_array(); v = context.get_full_volume_array()
        self._vidya = vidya(c, period=self.vidya_period, cmo_period=self.vidya_cmo)
        self._bb_u, self._bb_m, self._bb_l = bollinger_bands(c, period=20, std=2.0)
        self._atr = atr(h, l, c, period=14)
        self._avg_volume = fast_avg_volume(v, 20)

    def on_bar(self, context):
        i = context.bar_index; price = context.close_raw; side, lots = context.position
        if context.is_rollover: return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1: return
        av = self._atr[i]; vd = self._vidya[i]; bu = self._bb_u[i]; bl_v = self._bb_l[i]; bm = self._bb_m[i]
        if np.isnan(av) or av <= 0 or np.isnan(vd) or np.isnan(bu) or np.isnan(bl_v): return
        vd_prev = self._vidya[i - 1] if i > 0 else np.nan
        if np.isnan(vd_prev): return
        cp = context.get_full_close_array()[i]; vd_up = vd > vd_prev
        # BB width filter: not too wide (overextended) and not too narrow (squeeze)
        bb_width = (bu - bl_v) / bm if bm > 0 else 0
        self.bars_since_last_scale += 1

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            self.stop_price = max(self.stop_price, self.highest_since_entry - self.atr_stop_mult * av)
            if price <= self.stop_price: context.close_long(); self._reset_state(); return
        if side == 1 and self.entry_price > 0:
            pa = (price - self.entry_price) / av
            if pa >= 5.0 and not self._took_profit_5atr: context.close_long(lots=max(1, lots // 3)); self._took_profit_5atr = True; return
            elif pa >= 3.0 and not self._took_profit_3atr: context.close_long(lots=max(1, lots // 3)); self._took_profit_3atr = True; return
        if side == 1 and cp < vd: context.close_long(); self._reset_state(); return
        if side == 0 and cp > vd and vd_up and 0.02 < bb_width < 0.15:
            base = self._calc_lots(context, av)
            if base > 0:
                context.buy(base); self.entry_price = price; self.stop_price = price - self.atr_stop_mult * av
                self.highest_since_entry = price; self.position_scale = 1; self.bars_since_last_scale = 0
        elif side == 1 and self.position_scale < MAX_SCALE and self.bars_since_last_scale >= 10 and price > self.entry_price + av and vd_up:
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
