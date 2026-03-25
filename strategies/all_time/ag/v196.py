import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.kalman_trend import kalman_filter
from indicators.momentum.rsi import rsi
from indicators.volume.oi_divergence import oi_divergence
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV196(TimeSeriesStrategy):
    """
    策略简介：Kalman滤波趋势估计 + RSI回调入场 + OI背离确认的ML趋势策略。

    使用指标：
    - Kalman Filter(0.01, 1.0): 自适应趋势估计，slope>0上涨/<0下跌
    - RSI(14): 动量回调识别，趋势中回调到40-60区间入场
    - OI Divergence(20): 持仓量背离确认资金支持
    - ATR(14): 止损距离计算

    进场条件（做多）：Kalman slope>0 + RSI回调至30-50 + OI divergence>=0
    进场条件（做空）：Kalman slope<0 + RSI反弹至50-70 + OI divergence<=0

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - Kalman slope方向反转

    优点：Kalman自适应平滑噪音，RSI回调入场获得更好价位
    缺点：Kalman参数（process/measurement noise）敏感，OI可能滞后
    """
    name = "ag_alltime_v196"
    warmup = 500
    freq = "1h"

    kalman_pn: float = 0.01       # Optuna: 0.001-0.05
    kalman_mn: float = 1.0        # Optuna: 0.5-2.0
    rsi_period: int = 14          # Optuna: 10-20
    rsi_pullback_lo: float = 30.0 # Optuna: 25-40
    rsi_pullback_hi: float = 50.0 # Optuna: 45-55
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._kalman_level = None
        self._kalman_slope = None
        self._rsi = None
        self._oi_div = None
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

        self._kalman_level, self._kalman_slope, _ = kalman_filter(
            closes, self.kalman_pn, self.kalman_mn)
        self._rsi = rsi(closes, self.rsi_period)
        self._oi_div = oi_divergence(closes, oi, period=20)
        self._atr = atr(highs, lows, closes, period=14)

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
        slope = self._kalman_slope[i]
        rsi_val = self._rsi[i]
        oi_d = self._oi_div[i]

        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(slope) or np.isnan(rsi_val):
            return
        if np.isnan(oi_d):
            oi_d = 0.0

        prev_slope = self._kalman_slope[i - 1] if i > 0 else np.nan
        if np.isnan(prev_slope):
            return

        self.bars_since_last_scale += 1

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

        # 3. Signal exit: Kalman slope reversal
        if side == 1 and slope < 0 and prev_slope >= 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and slope > 0 and prev_slope <= 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: Kalman trend + RSI pullback + OI support
        if side == 0:
            if slope > 0 and self.rsi_pullback_lo <= rsi_val <= self.rsi_pullback_hi and oi_d >= 0:
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
            elif slope < 0 and (100 - self.rsi_pullback_hi) <= rsi_val <= (100 - self.rsi_pullback_lo) and oi_d <= 0:
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

        # 5. Scale-in
        elif side != 0 and self._should_add(price, atr_val):
            if self.direction == 1 and slope > 0:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and slope < 0:
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
