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
from indicators.volatility.bollinger import bollinger_bands
from indicators.volatility.keltner_width import keltner_width

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class MediumTrendV222(TimeSeriesStrategy):
    """
    策略简介：Bollinger突破 + Keltner Width收缩的4h做多策略。

    使用指标：
    - Bollinger Bands(20, 2.0): 价格波动带，突破上轨=强势
    - Keltner Width(20, 10, 1.5): Keltner通道宽度，收窄后突破更有效
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Keltner Width处于近20期低位（波动率压缩）
    - 价格突破Bollinger上轨（波动率扩张向上突破）

    出场条件：
    - ATR追踪止损（3.5倍ATR）
    - 分层止盈
    - 价格跌破Bollinger中轨

    优点：双通道确认波动率状态，突破信号高质量
    缺点：Keltner Width低位不一定导致方向性突破
    """
    name = "mt_v222"
    warmup = 60
    freq = "4h"

    atr_stop_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._bb_upper = None
        self._bb_mid = None
        self._kw = None
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
        self._bb_upper, self._bb_mid, _ = bollinger_bands(closes)
        self._kw = keltner_width(highs, lows, closes)
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
        bb_up = self._bb_upper[i]
        bb_m = self._bb_mid[i]
        kw = self._kw[i]
        if np.isnan(bb_up) or np.isnan(bb_m) or np.isnan(kw):
            return
        self.bars_since_last_scale += 1

        # Check if KW is in bottom 25% of recent 20 bars
        kw_low = False
        if i >= 20:
            kw_window = self._kw[i - 20:i]
            valid = kw_window[~np.isnan(kw_window)]
            if len(valid) > 5:
                kw_low = kw <= np.percentile(valid, 25)

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

        if side == 1 and price < bb_m:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and kw_low and price > bb_up:
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
