import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.aroon import aroon
from indicators.momentum.stoch_rsi import stoch_rsi
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV129(TimeSeriesStrategy):
    """
    策略简介：4h Aroon趋势方向 + 5min Stochastic RSI超卖入场。

    使用指标：
    - Aroon(25) [4h]: Aroon Up > 70 确认上升趋势
    - Stoch RSI(14,14,3,3) [5min]: %K超卖区域入场
    - ATR(14) [5min]: 止损距离

    进场条件（做多）：4h Aroon Up > 70 & Down < 30, 5min StochRSI %K < 20
    出场条件：ATR追踪止损, 分层止盈, Aroon Down > Aroon Up

    优点：Aroon识别新高趋势精准，StochRSI灵敏入场
    缺点：StochRSI波动剧烈，可能过早入场
    """
    name = "medium_trend_v129"
    freq = "5min"
    warmup = 2000

    aroon_period: int = 25
    aroon_up_threshold: float = 70.0
    stoch_rsi_entry: float = 20.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._stoch_k = None
        self._atr = None
        self._avg_volume = None
        self._aroon_up_4h = None
        self._aroon_down_4h = None
        self._4h_map = None

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

        k_arr, _ = stoch_rsi(closes, rsi=14, stoch=14, k=3, d=3)
        self._stoch_k = k_arr
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 48
        n_4h = n // step
        trim = n_4h * step
        highs_4h = highs[:trim].reshape(n_4h, step).max(axis=1)
        lows_4h = lows[:trim].reshape(n_4h, step).min(axis=1)

        self._aroon_up_4h, self._aroon_down_4h, _ = aroon(
            highs_4h, lows_4h, period=self.aroon_period)
        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1),
                                  len(self._aroon_up_4h) - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        sk = self._stoch_k[i]
        atr_val = self._atr[i]
        au = self._aroon_up_4h[j]
        ad = self._aroon_down_4h[j]
        if np.isnan(sk) or np.isnan(atr_val) or np.isnan(au) or np.isnan(ad):
            return

        uptrend = au > self.aroon_up_threshold and ad < 30
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

        if side == 1 and ad > au:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and uptrend and sk < self.stoch_rsi_entry:
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
                    and uptrend):
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
