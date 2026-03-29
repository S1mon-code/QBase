import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.volatility.atr import atr
from indicators.volume.obv import obv
from indicators.trend.ema import ema

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class MediumTrendV202(TimeSeriesStrategy):
    """
    策略简介：OBV趋势 + EMA方向确认的日线做多策略。

    使用指标：
    - OBV: 量价累积趋势，OBV上行表明买盘累积
    - EMA(20): 价格趋势方向过滤
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - 价格 > EMA(20)（价格处于上升趋势）
    - OBV创20日新高（量能持续流入）

    出场条件：
    - ATR追踪止损（3.5倍ATR）
    - 分层止盈（3ATR/5ATR）
    - 价格跌破EMA(20)

    优点：OBV领先于价格，提前捕捉资金流向变化
    缺点：OBV在低量市场信号噪声大
    """
    name = "mt_v202"
    warmup = 60
    freq = "daily"

    ema_period: int = 20
    obv_lookback: int = 20
    atr_stop_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._obv = None
        self._ema = None
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
        self._atr = atr(highs, lows, closes, period=14)
        self._obv = obv(closes, volumes)
        self._ema = ema(closes, self.ema_period)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return
        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        obv_val = self._obv[i]
        ema_val = self._ema[i]
        if np.isnan(obv_val) or np.isnan(ema_val):
            return
        self.bars_since_last_scale += 1

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

        if side == 1 and price < ema_val:
            context.close_long()
            self._reset_state()
            return

        # OBV 20-day high check
        obv_high = True
        if i >= self.obv_lookback:
            obv_window = self._obv[i - self.obv_lookback:i]
            if len(obv_window) > 0 and not np.all(np.isnan(obv_window)):
                obv_high = obv_val >= np.nanmax(obv_window)
            else:
                obv_high = False
        else:
            obv_high = False

        if side == 0 and price > ema_val and obv_high:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0
        elif side == 1 and self.position_scale < MAX_SCALE and self.bars_since_last_scale >= 10 and price > self.entry_price + atr_val:
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
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
