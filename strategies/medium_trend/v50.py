import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.donchian import donchian
from indicators.regime.efficiency_ratio import efficiency_ratio
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV50(TimeSeriesStrategy):
    """
    策略简介：Donchian Channel突破 + Efficiency Ratio过滤的30min通道策略。

    使用指标：
    - Donchian(20): 价格突破上轨为做多信号
    - Efficiency Ratio(20): >0.3时市场为趋势状态，适合突破交易
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - close >= Donchian上轨（新高突破）
    - ER > er_threshold（市场效率足够高，非震荡）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - close < Donchian中轨

    优点：Donchian是最纯粹的趋势突破，ER过滤震荡假突破
    缺点：Donchian在区间震荡中信号频繁，ER阈值需精心调整
    """
    name = "medium_trend_v50"
    warmup = 600
    freq = "30min"

    dc_period: int = 20               # Optuna: 15-40
    er_period: int = 20               # Optuna: 10-30
    er_threshold: float = 0.3         # Optuna: 0.2-0.5
    atr_stop_mult: float = 3.0       # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._dc_upper = None
        self._dc_mid = None
        self._er = None
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

        self._dc_upper, _, self._dc_mid = donchian(highs, lows, period=self.dc_period)
        self._er = efficiency_ratio(closes, period=self.er_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        close_val = context.get_full_close_array()[i]
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        dc_u = self._dc_upper[i]
        dc_m = self._dc_mid[i]
        er_val = self._er[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(dc_u) or np.isnan(er_val):
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

        if side == 1 and close_val < dc_m:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and close_val >= dc_u and er_val > self.er_threshold:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, er_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, er_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if er_val < self.er_threshold:
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
