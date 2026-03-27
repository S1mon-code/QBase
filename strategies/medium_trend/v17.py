import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.momentum.schaff_trend import schaff_trend_cycle
from indicators.trend.adx import adx
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV17(TimeSeriesStrategy):
    """
    策略简介：Schaff Trend Cycle动量 + ADX趋势强度过滤的做多策略（5min频率）。

    使用指标：
    - Schaff Trend Cycle(10,23,50): 趋势周期指标，>75为多头区
    - ADX(14): 趋势强度过滤，>20确认趋势存在
    - ATR(14): 止损距离计算

    进场条件（做多）：STC > 75 且 STC从低区上穿75 且 ADX > 20
    出场条件：ATR追踪止损 / 分层止盈 / STC < 25

    优点：STC比MACD更快速响应，ADX过滤震荡
    缺点：STC在高位可能持续很久导致信号不清晰
    """
    name = "mt_v17"
    warmup = 2000
    freq = "5min"

    stc_period: int = 10
    stc_fast: int = 23
    stc_slow: int = 50
    adx_period: int = 14
    adx_threshold: float = 20.0
    stc_buy: float = 75.0
    stc_sell: float = 25.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._stc = None
        self._adx = None
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

        self._stc = schaff_trend_cycle(closes, period=self.stc_period,
                                        fast=self.stc_fast, slow=self.stc_slow)
        self._adx = adx(highs, lows, closes, period=self.adx_period)
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
        stc_val = self._stc[i]
        adx_val = self._adx[i]
        if np.isnan(atr_val) or np.isnan(stc_val) or np.isnan(adx_val):
            return
        if i < 1 or np.isnan(self._stc[i - 1]):
            return

        self.bars_since_last_scale += 1
        prev_stc = self._stc[i - 1]
        stc_cross_up = prev_stc < self.stc_buy and stc_val >= self.stc_buy

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

        if side == 1 and stc_val < self.stc_sell:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and stc_cross_up and adx_val > self.adx_threshold:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, stc_val, adx_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, stc_val, adx_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if stc_val < self.stc_buy or adx_val < self.adx_threshold:
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
