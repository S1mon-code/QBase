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


def _kmeans_regime(closes, period=120):
    """K-Means-like regime: 0=sideways, 1=bullish, 2=bearish."""
    n = len(closes)
    cluster = np.full(n, 0, dtype=np.int32)
    if n < period + 1:
        return cluster
    log_ret = np.full(n, 0.0)
    log_ret[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-9))
    for i in range(period, n):
        w = log_ret[i - period + 1:i + 1]
        mu = np.mean(w)
        sigma = np.std(w)
        if sigma < 1e-12:
            continue
        cum = np.sum(w)
        thresh = sigma * np.sqrt(period) * 0.5
        if cum > thresh:
            cluster[i] = 1
        elif cum < -thresh:
            cluster[i] = 2
    return cluster


def _variance_ratio(closes, period=20, lag=5):
    """Variance ratio. VR>1=momentum, VR<1=mean-reversion."""
    n = len(closes)
    vr = np.full(n, np.nan, dtype=np.float64)
    if n < period + lag:
        return vr
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-9))
    log_ret_q = np.full(n, np.nan)
    for i in range(lag, n):
        log_ret_q[i] = np.sum(log_ret[i - lag + 1:i + 1])
    for i in range(period + lag, n):
        w1 = log_ret[i - period + 1:i + 1]
        v1 = w1[~np.isnan(w1)]
        wq = log_ret_q[i - period + 1:i + 1]
        vq = wq[~np.isnan(wq)]
        if len(v1) < 10 or len(vq) < 10:
            continue
        var1 = np.var(v1, ddof=1)
        varq = np.var(vq, ddof=1)
        if var1 > 1e-15:
            vr[i] = varq / (lag * var1)
    return vr


class StrategyV17(TimeSeriesStrategy):
    """
    策略简介：K-Means聚类 + 方差比率双重确认的动量策略。

    使用指标：
    - K-Means Regime(120): 聚类识别趋势状态
    - Variance Ratio(20,5): 确认动量存在(VR>1)
    - EMA(40): 方向过滤
    - ATR(14): 止损距离计算

    进场条件（做多）：聚类=1(看涨), VR>1.1(动量确认), 价格>EMA40
    进场条件（做空）：聚类=2(看跌), VR>1.1(动量确认), 价格<EMA40

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 聚类或VR反转

    优点：聚类识别宏观趋势+VR确认微观动量，双层验证
    缺点：两个指标都有滞后，在快速反转时反应慢
    """
    name = "ag_alltime_v17"
    warmup = 400
    freq = "daily"

    kmeans_period: int = 120      # Optuna: 60-200
    vr_period: int = 20           # Optuna: 15-40
    vr_lag: int = 5               # Optuna: 3-10
    vr_thresh: float = 1.1       # Optuna: 1.0-1.5
    ema_period: int = 40          # Optuna: 20-60
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._cluster = None
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

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._atr = atr(highs, lows, closes, period=14)
        self._ema = ema(closes, period=self.ema_period)
        self._cluster = _kmeans_regime(closes, period=self.kmeans_period)
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
        cluster = self._cluster[i]
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
        if side == 1 and (cluster == 2 or vr_val < 0.9):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (cluster == 1 or vr_val < 0.9):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry: both K-Means and VR must confirm
        if side == 0:
            if cluster == 1 and vr_val > self.vr_thresh and price > ema_val:
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
            elif cluster == 2 and vr_val > self.vr_thresh and price < ema_val:
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
            if self.direction == 1 and cluster == 1 and vr_val > self.vr_thresh:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and cluster == 2 and vr_val > self.vr_thresh:
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
