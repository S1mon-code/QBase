import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.adx import adx
from indicators.volume.obv import obv
from indicators.trend.ema import ema
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV138(TimeSeriesStrategy):
    """
    策略简介：日线ADX趋势强度 + 30min OBV突破入场的多周期策略。

    使用指标：
    - ADX(14) [日线]: 趋势强度确认
    - OBV [30min]: OBV > OBV EMA(20) 确认量能支持上涨
    - ATR(14) [30min]: 止损距离

    进场条件（做多）：日线ADX > threshold, 30min OBV > OBV_EMA且价格上涨
    出场条件：ATR追踪止损, 分层止盈, ADX下降

    优点：OBV领先价格确认买盘，ADX过滤震荡
    缺点：OBV在跳空环境下可能失真
    """
    name = "medium_trend_v138"
    freq = "30min"
    warmup = 1000

    adx_period: int = 14
    adx_threshold: float = 25.0
    obv_ema_period: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._obv = None
        self._obv_ema = None
        self._atr = None
        self._avg_volume = None
        self._adx_d = None
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

        self._obv = obv(closes, volumes)
        self._obv_ema = ema(self._obv, period=self.obv_ema_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 8  # 30min * 8 = 4h ~ daily
        n_d = n // step
        trim = n_d * step
        closes_d = closes[:trim].reshape(n_d, step)[:, -1]
        highs_d = highs[:trim].reshape(n_d, step).max(axis=1)
        lows_d = lows[:trim].reshape(n_d, step).min(axis=1)

        self._adx_d = adx(highs_d, lows_d, closes_d, period=self.adx_period)
        self._daily_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1),
                                     len(self._adx_d) - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._daily_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        obv_val = self._obv[i]
        obv_ema_val = self._obv_ema[i]
        atr_val = self._atr[i]
        adx_val = self._adx_d[j]
        if np.isnan(obv_val) or np.isnan(obv_ema_val) or np.isnan(atr_val) or np.isnan(adx_val):
            return

        trending = adx_val > self.adx_threshold
        obv_bullish = obv_val > obv_ema_val
        price_rising = price > context.open_raw
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

        if side == 1 and adx_val < self.adx_threshold * 0.6:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and trending and obv_bullish and price_rising:
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
                    and trending and obv_bullish):
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
