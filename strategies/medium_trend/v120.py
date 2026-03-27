import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.structure.smart_money import smart_money_index
from indicators.trend.hma import hma
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV120(TimeSeriesStrategy):
    """
    策略简介：Smart Money Index聪明钱方向 + HMA趋势确认的日线多头策略。

    使用指标：
    - Smart Money Index(20): 聪明钱指标上升确认机构资金流入
    - HMA(20): Hull均线上升确认趋势
    - ATR(14): 止损

    进场：SMI上升 + HMA上升 + 价格>HMA
    出场：HMA下降 / 价格<HMA / ATR止损 / 分层止盈

    优点：SMI捕捉机构行为，HMA低滞后趋势确认
    缺点：SMI在非交易时段数据有限，代理指标精度有限
    """
    name = "medium_trend_v120"
    warmup = 80
    freq = "daily"

    smi_period: int = 20; hma_period: int = 20; smi_lookback: int = 5
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._smi = None; self._hma = None; self._atr = None; self._avg_volume = None

    def on_init(self, context):
        self.entry_price = 0.0; self.stop_price = 0.0; self.highest_since_entry = 0.0
        self.position_scale = 0; self.bars_since_last_scale = 0
        self._took_profit_3atr = False; self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        c = context.get_full_close_array(); h = context.get_full_high_array()
        l = context.get_full_low_array(); v = context.get_full_volume_array()
        o = context.get_full_open_array()
        self._smi = smart_money_index(o, c, h, l, v, period=self.smi_period)
        self._hma = hma(c, self.hma_period)
        self._atr = atr(h, l, c, period=14)
        self._avg_volume = fast_avg_volume(v, 20)

    def on_bar(self, context):
        i = context.bar_index; price = context.close_raw; side, lots = context.position
        if context.is_rollover: return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1: return
        av = self._atr[i]; sm = self._smi[i]; hv = self._hma[i]
        if np.isnan(av) or av <= 0 or np.isnan(sm) or np.isnan(hv): return
        sm_prev = self._smi[i - self.smi_lookback] if i >= self.smi_lookback else np.nan
        hv_prev = self._hma[i - 2] if i >= 2 else np.nan
        if np.isnan(sm_prev) or np.isnan(hv_prev): return
        cp = context.get_full_close_array()[i]
        smi_up = sm > sm_prev; hma_up = hv > hv_prev
        self.bars_since_last_scale += 1

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            self.stop_price = max(self.stop_price, self.highest_since_entry - self.atr_stop_mult * av)
            if price <= self.stop_price: context.close_long(); self._reset_state(); return
        if side == 1 and self.entry_price > 0:
            pa = (price - self.entry_price) / av
            if pa >= 5.0 and not self._took_profit_5atr: context.close_long(lots=max(1, lots // 3)); self._took_profit_5atr = True; return
            elif pa >= 3.0 and not self._took_profit_3atr: context.close_long(lots=max(1, lots // 3)); self._took_profit_3atr = True; return
        if side == 1 and (not hma_up or cp < hv): context.close_long(); self._reset_state(); return
        if side == 0 and smi_up and hma_up and cp > hv:
            bl = self._calc_lots(context, av)
            if bl > 0:
                context.buy(bl); self.entry_price = price; self.stop_price = price - self.atr_stop_mult * av
                self.highest_since_entry = price; self.position_scale = 1; self.bars_since_last_scale = 0
        elif side == 1 and self.position_scale < MAX_SCALE and self.bars_since_last_scale >= 10 and price > self.entry_price + av and hma_up:
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
