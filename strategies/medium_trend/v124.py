import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.trend.ema import ema_cross
from indicators.volatility.bollinger import bollinger_bands
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV124(TimeSeriesStrategy):
    """
    策略简介：日线EMA金叉定方向 + 30min布林带下轨回弹入场的多周期策略。

    使用指标：
    - EMA Cross(20, 60) [日线]: 金叉确认上升趋势
    - Bollinger Bands(20, 2.0) [30min]: 下轨附近寻找回调入场
    - ATR(14) [30min]: 止损距离计算

    进场条件（做多）：
    - 日线 EMA 快线 > 慢线（上升趋势）
    - 30min 价格触及布林带下轨附近（close < lower + 0.3 * bandwidth）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR / 5ATR）
    - 日线 EMA 死叉

    优点：布林带入场点回撤小，EMA过滤确保顺势
    缺点：布林带在强趋势中下轨可能长期不触及
    """
    name = "medium_trend_v124"
    freq = "30min"
    warmup = 1000

    ema_fast: int = 20
    ema_slow: int = 60
    bb_period: int = 20
    bb_std: float = 2.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._bb_upper = None
        self._bb_mid = None
        self._bb_lower = None
        self._atr = None
        self._avg_volume = None
        self._ema_fast_d = None
        self._ema_slow_d = None
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

        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            closes, period=self.bb_period, std=self.bb_std)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 8  # 30min * 8 = 4h; daily ~= step*N
        # For daily from 30min: Chinese futures ~8 x 30min = 4h/day
        # Use 8 for approximate daily
        n_d = n // step
        trim = n_d * step
        closes_d = closes[:trim].reshape(n_d, step)[:, -1]

        self._ema_fast_d, self._ema_slow_d, _ = ema_cross(
            closes_d, fast_period=self.ema_fast, slow_period=self.ema_slow)
        self._daily_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1),
                                     len(self._ema_fast_d) - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._daily_map[i]
        price = context.close_raw
        close = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        bb_lower = self._bb_lower[i]
        bb_upper = self._bb_upper[i]
        ema_f = self._ema_fast_d[j]
        ema_s = self._ema_slow_d[j]
        if np.isnan(atr_val) or np.isnan(bb_lower) or np.isnan(ema_f) or np.isnan(ema_s):
            return

        uptrend = ema_f > ema_s
        bandwidth = bb_upper - bb_lower
        near_lower = close < bb_lower + 0.3 * bandwidth if bandwidth > 0 else False

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

        if side == 1 and not uptrend:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and uptrend and near_lower:
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
