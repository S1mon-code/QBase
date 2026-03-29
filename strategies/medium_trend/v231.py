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
from indicators.momentum.macd import macd
from indicators.momentum.rsi import rsi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class MediumTrendV231(TimeSeriesStrategy):
    """
    策略简介：MACD金叉 + RSI中性区确认的1h做多策略。

    使用指标：
    - MACD(12, 26, 9): 经典趋势动量指标
    - RSI(14): 相对强弱指标，过滤超买/超卖
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - MACD线上穿Signal线（金叉）
    - RSI在40-70之间（不超卖也不超买，健康趋势区间）

    出场条件：
    - ATR追踪止损（3.0倍ATR）
    - 分层止盈
    - MACD线下穿Signal线（死叉）

    优点：MACD+RSI经典组合，互补性强
    缺点：震荡市金叉死叉频繁
    """
    name = "mt_v231"
    warmup = 60
    freq = "1h"

    rsi_low: float = 40.0
    rsi_high: float = 70.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._macd_line = None
        self._macd_sig = None
        self._rsi = None
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
        self._macd_line, self._macd_sig, _ = macd(closes)
        self._rsi = rsi(closes)
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
        ml = self._macd_line[i]
        ms = self._macd_sig[i]
        rsi_val = self._rsi[i]
        if np.isnan(ml) or np.isnan(ms) or np.isnan(rsi_val):
            return
        prev_ml = self._macd_line[i - 1] if i > 0 else np.nan
        prev_ms = self._macd_sig[i - 1] if i > 0 else np.nan
        if np.isnan(prev_ml) or np.isnan(prev_ms):
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

        if side == 1 and ml < ms:
            context.close_long()
            self._reset_state()
            return

        # Golden cross with RSI filter
        if side == 0 and prev_ml <= prev_ms and ml > ms and self.rsi_low < rsi_val < self.rsi_high:
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
