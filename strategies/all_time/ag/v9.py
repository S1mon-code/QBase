import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.trend.ema import ema

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _variance_ratio(closes, period=20, lag=5):
    """Lo-MacKinlay Variance Ratio test (rolling).

    VR(q) = Var(q-period returns) / (q * Var(1-period returns))
    VR > 1 = momentum (positive autocorrelation)
    VR < 1 = mean-reversion (negative autocorrelation)
    VR = 1 = random walk
    """
    n = len(closes)
    vr = np.full(n, np.nan, dtype=np.float64)

    if n < period + lag:
        return vr

    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-9))

    # Multi-period returns
    log_ret_q = np.full(n, np.nan)
    for i in range(lag, n):
        log_ret_q[i] = np.sum(log_ret[i - lag + 1:i + 1])

    for i in range(period + lag, n):
        # 1-period variance
        window1 = log_ret[i - period + 1:i + 1]
        valid1 = window1[~np.isnan(window1)]
        if len(valid1) < 10:
            continue
        var1 = np.var(valid1, ddof=1)

        # q-period variance
        windowq = log_ret_q[i - period + 1:i + 1]
        validq = windowq[~np.isnan(windowq)]
        if len(validq) < 10:
            continue
        varq = np.var(validq, ddof=1)

        if var1 > 1e-15:
            vr[i] = varq / (lag * var1)

    return vr


class StrategyV9(TimeSeriesStrategy):
    """
    策略简介：基于方差比率判断动量/均值回归，自适应交易方向。

    使用指标：
    - Variance Ratio(20, lag=5): VR>1动量跟踪，VR<1均值回归
    - EMA(30): 趋势方向
    - ATR(14): 止损距离计算

    进场条件（做多-动量）：VR>1.2, 价格>EMA30
    进场条件（做空-动量）：VR>1.2, 价格<EMA30
    进场条件（做多-回归）：VR<0.8, 价格<EMA30-1.5ATR
    进场条件（做空-回归）：VR<0.8, 价格>EMA30+1.5ATR

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - VR回到中性区间[0.8,1.2]

    优点：统计检验基础扎实，自适应动量/回归模式
    缺点：VR估计噪声大，需要足够数据窗口
    """
    name = "ag_alltime_v9"
    warmup = 250
    freq = "daily"

    vr_period: int = 20           # Optuna: 15-40
    vr_lag: int = 5               # Optuna: 3-10
    momentum_thresh: float = 1.2  # Optuna: 1.1-1.5
    revert_thresh: float = 0.8   # Optuna: 0.5-0.9
    ema_period: int = 30          # Optuna: 15-50
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._vr = None
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
        self.mode = 0  # 1=momentum, 2=mean_revert

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._atr = atr(highs, lows, closes, period=14)
        self._ema = ema(closes, period=self.ema_period)
        self._vr = _variance_ratio(closes, period=self.vr_period, lag=self.vr_lag)

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
        vr_val = self._vr[i]
        ema_val = self._ema[i]
        if np.isnan(vr_val) or np.isnan(ema_val):
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
        # VR back to neutral zone means exit
        if side != 0 and self.revert_thresh <= vr_val <= self.momentum_thresh:
            if side == 1:
                context.close_long()
            else:
                context.close_short()
            self._reset_state()
            return
        # Mean-revert mode: take profit at EMA
        if self.mode == 2 and side != 0:
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
            if vr_val > self.momentum_thresh:
                # Momentum mode
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
                else:
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
            elif vr_val < self.revert_thresh:
                # Mean-reversion mode
                if price < ema_val - 1.5 * atr_val:
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
                elif price > ema_val + 1.5 * atr_val:
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

        # 5. Scale-in (momentum mode only)
        elif side != 0 and self.mode == 1 and self._should_add(price, atr_val):
            if self.direction == 1 and vr_val > self.momentum_thresh:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and vr_val > self.momentum_thresh:
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
