import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.trend.adx import adx_with_di
from indicators.volatility.historical_vol import historical_volatility

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _kalman_trend(closes, process_noise=0.01, measurement_noise=1.0):
    """1D Kalman filter. Returns (filtered_price, slope)."""
    n = len(closes)
    filtered = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    if n < 2:
        return filtered, slope
    x = np.array([closes[0], 0.0])
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[process_noise, 0.0], [0.0, process_noise]])
    R = np.array([[measurement_noise]])
    for i in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        z = closes[i]
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T / S[0, 0]
        x = x_pred + K.flatten() * y[0]
        P = (np.eye(2) - K @ H) @ P_pred
        filtered[i] = x[0]
        slope[i] = x[1]
    return filtered, slope


def _market_state(adx_arr, hvol_arr, closes, period=20):
    """Market state: 1=trend_up, 2=trend_down, 0=range, 3=volatile."""
    n = len(closes)
    state = np.full(n, 0, dtype=np.int32)
    for i in range(period, n):
        a = adx_arr[i]
        hv = hvol_arr[i]
        if np.isnan(a) or np.isnan(hv):
            continue
        price_dir = 1 if closes[i] > closes[i - period] else (-1 if closes[i] < closes[i - period] else 0)
        hv_window = hvol_arr[max(0, i - 120):i + 1]
        hv_valid = hv_window[~np.isnan(hv_window)]
        if len(hv_valid) < 10:
            continue
        hv_pct = np.sum(hv_valid < hv) / len(hv_valid)
        if a > 25:
            state[i] = 1 if price_dir > 0 else 2
        elif hv_pct > 0.8:
            state[i] = 3
        else:
            state[i] = 0
    return state


class StrategyV18(TimeSeriesStrategy):
    """
    策略简介：Kalman滤波方向 + 市场状态过滤的高频趋势策略。

    使用指标：
    - Kalman Filter: 平滑价格提取斜率方向
    - Market State (ADX+HV): 过滤非趋势状态
    - ATR(14): 止损距离计算

    进场条件（做多）：Kalman斜率>0.3, 市场状态=trend_up(1)
    进场条件（做空）：Kalman斜率<-0.3, 市场状态=trend_down(2)

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 市场状态切换为非趋势或Kalman斜率反转

    优点：Kalman低延迟+状态过滤减少震荡市交易
    缺点：1h频率交易成本较高，快速状态切换时可能来回止损
    """
    name = "ag_alltime_v18"
    warmup = 500
    freq = "1h"

    process_noise: float = 0.01    # Optuna: 0.001-0.1
    measurement_noise: float = 1.0  # Optuna: 0.1-5.0
    slope_thresh: float = 0.3     # Optuna: 0.1-1.0
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._kalman_price = None
        self._kalman_slope = None
        self._mkt_state = None
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

        self._atr = atr(highs, lows, closes, period=14)
        self._kalman_price, self._kalman_slope = _kalman_trend(
            closes, self.process_noise, self.measurement_noise
        )
        adx_arr, _, _ = adx_with_di(highs, lows, closes, period=14)
        hvol_arr = historical_volatility(closes, period=20)
        self._mkt_state = _market_state(adx_arr, hvol_arr, closes)

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
        kf_slope = self._kalman_slope[i]
        mkt = self._mkt_state[i]
        if np.isnan(kf_slope):
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

        # 3. Signal-based exit
        if side == 1 and (kf_slope < 0 or mkt in (0, 2, 3)):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (kf_slope > 0 or mkt in (0, 1, 3)):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry: Kalman + market state must agree
        if side == 0:
            if kf_slope > self.slope_thresh and mkt == 1:
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
            elif kf_slope < -self.slope_thresh and mkt == 2:
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
            if self.direction == 1 and kf_slope > self.slope_thresh and mkt == 1:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and kf_slope < -self.slope_thresh and mkt == 2:
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
