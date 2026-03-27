import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.trend.ema import ema

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _fractal_dimension(closes, period=30):
    """Rolling fractal dimension via box-counting approximation.

    D close to 1.0 = trending (smooth), D close to 2.0 = mean-reverting (rough).
    Uses Higuchi's method approximation.
    """
    n = len(closes)
    fd = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return fd

    for i in range(period, n):
        window = closes[i - period + 1:i + 1]
        # Higuchi-like: measure path length at different scales
        lengths = []
        scales = [1, 2, 4, 8]
        for k in scales:
            if k >= period // 2:
                continue
            num_segments = (period - 1) // k
            if num_segments < 1:
                continue
            path = 0.0
            for m in range(k):
                idx_arr = list(range(m, period, k))
                if len(idx_arr) < 2:
                    continue
                for j in range(1, len(idx_arr)):
                    path += abs(window[idx_arr[j]] - window[idx_arr[j - 1]])
            norm_length = path * (period - 1) / (k * k * num_segments) if num_segments > 0 else 0
            if norm_length > 0:
                lengths.append((np.log(1.0 / k), np.log(norm_length)))

        if len(lengths) >= 2:
            x = np.array([l[0] for l in lengths])
            y = np.array([l[1] for l in lengths])
            slope = np.polyfit(x, y, 1)[0]
            fd[i] = np.clip(slope, 1.0, 2.0)
        else:
            fd[i] = 1.5

    return fd


class StrategyV6(TimeSeriesStrategy):
    """
    策略简介：基于分形维度区分趋势/震荡，趋势时跟踪震荡时反转。

    使用指标：
    - Fractal Dimension(30): D<1.4趋势跟踪，D>1.6均值回归
    - EMA(20): 趋势方向判断
    - ATR(14): 止损距离计算

    进场条件（做多-趋势）：FD<1.4, 价格>EMA20
    进场条件（做空-趋势）：FD<1.4, 价格<EMA20
    进场条件（做多-回归）：FD>1.6, 价格<EMA20-1.5ATR (超卖反弹)
    进场条件（做空-回归）：FD>1.6, 价格>EMA20+1.5ATR (超买回落)

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - FD状态切换

    优点：自适应在趋势和震荡之间切换策略模式
    缺点：FD计算有滞后，边界值附近策略切换频繁
    """
    name = "ag_alltime_v6"
    warmup = 200
    freq = "daily"

    fd_period: int = 30           # Optuna: 20-60
    ema_period: int = 20          # Optuna: 10-40
    trend_fd: float = 1.4        # Optuna: 1.2-1.5
    revert_fd: float = 1.6       # Optuna: 1.5-1.8
    revert_atr_mult: float = 1.5  # Optuna: 1.0-2.5
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._fd = None
        self._ema = None
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
        self.mode = 0  # 0=none, 1=trend, 2=mean_revert

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._atr = atr(highs, lows, closes, period=14)
        self._ema = ema(closes, period=self.ema_period)
        self._fd = _fractal_dimension(closes, period=self.fd_period)

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

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
        fd_val = self._fd[i]
        ema_val = self._ema[i]
        if np.isnan(fd_val) or np.isnan(ema_val):
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

        # 3. Signal-based exit: mode mismatch
        if side != 0:
            # If we entered in trend mode but FD says mean-revert (or vice versa)
            if self.mode == 1 and fd_val > self.revert_fd:
                if side == 1:
                    context.close_long()
                else:
                    context.close_short()
                self._reset_state()
                return
            if self.mode == 2 and fd_val < self.trend_fd:
                if side == 1:
                    context.close_long()
                else:
                    context.close_short()
                self._reset_state()
                return
            # Mean-revert profit target: revert to EMA
            if self.mode == 2:
                if side == 1 and price >= ema_val:
                    context.close_long()
                    self._reset_state()
                    return
                if side == -1 and price <= ema_val:
                    context.close_short()
                    self._reset_state()
                    return

        # 4. Entry
        if side == 0:
            if fd_val < self.trend_fd:
                # Trend following mode
                if price > ema_val:
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
                        self.mode = 1
                elif price < ema_val:
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
                        self.mode = 1
            elif fd_val > self.revert_fd:
                # Mean-reversion mode
                if price < ema_val - self.revert_atr_mult * atr_val:
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
                        self.mode = 2
                elif price > ema_val + self.revert_atr_mult * atr_val:
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
                        self.mode = 2

        # 5. Scale-in (trend mode only)
        elif side != 0 and self.mode == 1 and self._should_add(price, atr_val):
            if self.direction == 1 and fd_val < self.trend_fd:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and fd_val < self.trend_fd:
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
        self.mode = 0
