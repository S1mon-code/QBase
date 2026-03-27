import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.ema import ema_cross
from indicators.regime.trend_strength_composite import trend_strength
from indicators.momentum.cmo import cmo
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV147(TimeSeriesStrategy):
    """
    策略简介：日线EMA金叉 + 4h Trend Strength + 30min CMO三周期策略。

    使用指标：
    - EMA Cross(20,60) [日线]: 金叉确认大趋势
    - Trend Strength(20) [4h]: 中周期趋势强度>60
    - CMO(14) [30min]: 动量>0确认入场
    - ATR(14) [30min]: 止损距离

    进场条件（做多）：日线金叉 + 4h TS>60 + 30min CMO>10
    出场条件：ATR追踪止损, 分层止盈, EMA死叉

    优点：三层逐级缩小确认，过滤效果极佳
    缺点：EMA金叉信号本身滞后
    """
    name = "medium_trend_v147"
    freq = "30min"
    warmup = 1500

    ts_threshold: float = 60.0
    cmo_threshold: float = 10.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._cmo = None
        self._atr = None
        self._avg_volume = None
        self._ema_f_d = None
        self._ema_s_d = None
        self._ts_4h = None
        self._d_map = None
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

        self._cmo = cmo(closes, period=14)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        # 4h from 30min (step=8)
        step_4h = 8
        n_4h = n // step_4h
        trim_4h = n_4h * step_4h
        closes_4h = closes[:trim_4h].reshape(n_4h, step_4h)[:, -1]
        highs_4h = highs[:trim_4h].reshape(n_4h, step_4h).max(axis=1)
        lows_4h = lows[:trim_4h].reshape(n_4h, step_4h).min(axis=1)
        self._ts_4h = trend_strength(closes_4h, highs_4h, lows_4h, period=20)
        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step_4h - 1), n_4h - 1)

        # Daily ~ 4h for Chinese futures
        self._ema_f_d, self._ema_s_d, _ = ema_cross(closes_4h, fast_period=20, slow_period=60)
        self._d_map = self._4h_map

    def on_bar(self, context):
        i = context.bar_index
        j4 = self._4h_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        cmo_val = self._cmo[i]
        atr_val = self._atr[i]
        ef = self._ema_f_d[j4]
        es = self._ema_s_d[j4]
        ts = self._ts_4h[j4]
        if np.isnan(cmo_val) or np.isnan(atr_val) or np.isnan(ef) or np.isnan(es) or np.isnan(ts):
            return

        golden = ef > es
        strong_trend = ts > self.ts_threshold
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

        if side == 1 and not golden:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and golden and strong_trend and cmo_val > self.cmo_threshold:
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
                    and golden and strong_trend):
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
