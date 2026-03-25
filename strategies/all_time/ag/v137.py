import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.kalman_trend import kalman_filter
from indicators.regime.market_state import market_state
from indicators.volatility.nr7 import nr7

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV137(TimeSeriesStrategy):
    """
    策略简介：Kalman滤波趋势方向 + 多维市场状态 + NR7窄幅突破的多空策略。

    使用指标：
    - Kalman Filter(0.01, 1.0): 卡尔曼滤波斜率作为趋势方向，正=上涨，负=下跌
    - Market State(20): 五状态分类器(quiet/up/down/volatile/breakout)，趋势态过滤
    - NR7: 7日最窄范围检测，波动率收缩后入场

    进场条件（做多）：NR7触发 + 市场状态=trending_up(1) + Kalman斜率>0
    进场条件（做空）：NR7触发 + 市场状态=trending_down(2) + Kalman斜率<0

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 市场状态变为quiet(0)或volatile_range(3)

    优点：NR7精确捕捉波动率收缩后爆发点，Kalman平滑过滤噪声
    缺点：NR7触发稀疏，可能长时间不交易；Kalman对剧烈波动反应迟钝
    """
    name = "ag_alltime_v137"
    warmup = 400
    freq = "daily"

    kalman_q: float = 0.01        # Optuna: 0.001-0.05
    kalman_r: float = 1.0         # Optuna: 0.5-5.0
    ms_period: int = 20           # Optuna: 10-40
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._kalman_slope = None
        self._market_state = None
        self._ms_confidence = None
        self._nr7 = None
        self._atr = None
        self._avg_volume = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.direction = 0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()

        self._atr = atr(highs, lows, closes, period=14)

        # Kalman filter
        _, self._kalman_slope, _ = kalman_filter(
            closes, process_noise=self.kalman_q, measurement_noise=self.kalman_r)

        # Market state
        self._market_state, self._ms_confidence = market_state(
            closes, volumes, oi, period=self.ms_period)

        # NR7
        self._nr7 = nr7(highs, lows)

        window = 20
        self._avg_volume = np.full_like(volumes, np.nan)
        for idx in range(window, len(volumes)):
            self._avg_volume[idx] = np.mean(volumes[idx - window:idx])

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        slope = self._kalman_slope[i]
        ms_val = self._market_state[i]
        nr7_val = self._nr7[i]
        if np.isnan(slope) or np.isnan(ms_val):
            return

        self.bars_since_last_scale += 1

        ms_int = int(ms_val)
        is_nr7 = bool(nr7_val) if not np.isnan(nr7_val) else False

        # 1. Stop loss
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return
        elif side == -1:
            self.lowest_since_entry = min(self.lowest_since_entry, price)
            trailing = self.lowest_since_entry + self.atr_stop_mult * atr_val
            self.stop_price = min(self.stop_price, trailing)
            if price >= self.stop_price:
                context.close_short()
                self._reset_state()
                return

        # 2. Tiered profit-taking
        if side != 0 and self.entry_price > 0:
            if side == 1:
                profit_atr = (price - self.entry_price) / atr_val
            else:
                profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_3atr = True
                return

        # 3. Signal-based exit: market state quiet or volatile range
        if side == 1 and (ms_int == 0 or ms_int == 3 or slope < 0):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (ms_int == 0 or ms_int == 3 or slope > 0):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry: NR7 trigger in trending state with Kalman direction
        if side == 0 and is_nr7:
            if ms_int == 1 and slope > 0:  # trending up
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
                    self.direction = 1
            elif ms_int == 2 and slope < 0:  # trending down
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
                    self.direction = -1

        # 5. Scale-in (use market state confirmation, no NR7 required)
        elif side != 0 and self._should_add(price, atr_val):
            if self.direction == 1 and ms_int == 1 and slope > 0:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and ms_int == 2 and slope < 0:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.sell(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1 and price < self.entry_price + atr_val:
            return False
        if self.direction == -1 and price > self.entry_price - atr_val:
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
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.direction = 0
