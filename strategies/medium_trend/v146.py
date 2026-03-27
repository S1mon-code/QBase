import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.trend.adx import adx
from indicators.trend.supertrend import supertrend
from indicators.momentum.stoch_rsi import stoch_rsi
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV146(TimeSeriesStrategy):
    """
    策略简介：日线ADX + 4h Supertrend + 5min Stochastic RSI三周期策略。

    使用指标：
    - ADX(14) [日线]: 趋势强度 > 25
    - Supertrend(10, 3.0) [4h]: 趋势方向过滤
    - Stoch RSI(14,14,3,3) [5min]: 超卖入场
    - ATR(14) [5min]: 止损距离

    进场条件（做多）：日线ADX>25 + 4h ST=1 + 5min StochRSI %K<20
    出场条件：ATR追踪止损, 分层止盈, ST翻转

    优点：ADX过滤震荡+Supertrend方向+精确入场
    缺点：ADX滞后，可能错过趋势初期
    """
    name = "medium_trend_v146"
    freq = "5min"
    warmup = 3000

    adx_threshold: float = 25.0
    stoch_entry: float = 20.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._stoch_k = None
        self._atr = None
        self._avg_volume = None
        self._adx_d = None
        self._st_dir = None
        self._map = None

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
        nd = n // step
        trim = nd * step
        closes_d = closes[:trim].reshape(nd, step)[:, -1]
        highs_d = highs[:trim].reshape(nd, step).max(axis=1)
        lows_d = lows[:trim].reshape(nd, step).min(axis=1)

        self._adx_d = adx(highs_d, lows_d, closes_d, period=14)
        _, self._st_dir = supertrend(highs_d, lows_d, closes_d, period=10, multiplier=3.0)
        self._map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1), nd - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._map[i]
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        sk = self._stoch_k[i]
        atr_val = self._atr[i]
        adx_val = self._adx_d[j]
        sd = self._st_dir[j]
        if np.isnan(sk) or np.isnan(atr_val) or np.isnan(adx_val) or np.isnan(sd):
            return

        trending = adx_val > self.adx_threshold
        daily_up = sd == 1
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

        if side == 1 and not daily_up:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and trending and daily_up and sk < self.stoch_entry:
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
                    and trending and daily_up):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
