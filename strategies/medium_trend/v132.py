import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.keltner import keltner
from indicators.momentum.cmo import cmo
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV132(TimeSeriesStrategy):
    """
    策略简介：日线Keltner通道趋势 + 10min CMO动量入场的多周期策略。

    使用指标：
    - Keltner(20, 10, 1.5) [日线]: 价格在上轨上方=强势上升
    - CMO(14) [10min]: Chande Momentum > 0 确认做多动量
    - ATR(14) [10min]: 止损距离

    进场条件（做多）：日线close > Keltner上轨, 10min CMO > cmo_threshold
    出场条件：ATR追踪止损, 分层止盈, close < Keltner中轨

    优点：Keltner突破过滤弱势行情，CMO双向动量确认
    缺点：突破入场可能追高
    """
    name = "medium_trend_v132"
    freq = "10min"
    warmup = 2000

    cmo_period: int = 14
    cmo_threshold: float = 10.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._cmo = None
        self._atr = None
        self._avg_volume = None
        self._kc_upper_d = None
        self._kc_mid_d = None
        self._closes_d = None
        self._daily_map = None

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
        n = len(closes)

        self._cmo = cmo(closes, period=self.cmo_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 24  # 10min * 24 = 4h ~ daily
        n_d = n // step
        trim = n_d * step
        closes_d = closes[:trim].reshape(n_d, step)[:, -1]
        highs_d = highs[:trim].reshape(n_d, step).max(axis=1)
        lows_d = lows[:trim].reshape(n_d, step).min(axis=1)

        self._kc_upper_d, self._kc_mid_d, _ = keltner(
            highs_d, lows_d, closes_d, ema=20, atr=10, mult=1.5)
        self._closes_d = closes_d
        self._daily_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1),
                                     len(closes_d) - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._daily_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        cmo_val = self._cmo[i]
        atr_val = self._atr[i]
        kc_upper = self._kc_upper_d[j]
        kc_mid = self._kc_mid_d[j]
        cd = self._closes_d[j]
        if np.isnan(cmo_val) or np.isnan(atr_val) or np.isnan(kc_upper):
            return

        above_upper = cd > kc_upper
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

        if side == 1 and not np.isnan(kc_mid) and cd < kc_mid:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and above_upper and cmo_val > self.cmo_threshold:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and above_upper and cmo_val > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
