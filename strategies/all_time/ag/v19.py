import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi
from indicators.momentum.roc import rate_of_change
from indicators.trend.ema import ema
from indicators.trend.linear_regression import linear_regression_slope

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _boosting_signal(features_list, closes, lookback=60):
    """Boosting ensemble signal. Returns signal in [-1, 1]."""
    n = len(closes)
    signal = np.full(n, np.nan, dtype=np.float64)
    if n < lookback + 1:
        return signal
    fwd_ret = np.full(n, np.nan)
    fwd_ret[:-1] = closes[1:] / closes[:-1] - 1.0
    n_f = len(features_list)
    for i in range(lookback, n):
        votes = np.zeros(n_f)
        weights = np.ones(n_f)
        for f_idx, feat in enumerate(features_list):
            val = feat[i]
            if np.isnan(val):
                continue
            w = feat[i - lookback:i]
            v = w[~np.isnan(w)]
            if len(v) < 10:
                continue
            z = (val - np.mean(v)) / max(np.std(v), 1e-12)
            if z > 0.5:
                votes[f_idx] = 1.0
            elif z < -0.5:
                votes[f_idx] = -1.0
            correct = 0
            total = 0
            for j in range(max(lookback, i - 20), i):
                fj = feat[j]
                if np.isnan(fj) or np.isnan(fwd_ret[j]):
                    continue
                wj = feat[j - lookback:j]
                vj = wj[~np.isnan(wj)]
                if len(vj) < 10:
                    continue
                zj = (fj - np.mean(vj)) / max(np.std(vj), 1e-12)
                pred = 1 if zj > 0 else -1
                actual = 1 if fwd_ret[j] > 0 else -1
                if pred == actual:
                    correct += 1
                total += 1
            if total > 5:
                weights[f_idx] = max(0.1, correct / total)
        tw = np.sum(np.abs(weights))
        if tw > 0:
            signal[i] = np.sum(votes * weights) / tw
    return signal


def _fractal_dimension(closes, period=30):
    """Rolling fractal dimension. D~1=trending, D~2=mean-reverting."""
    n = len(closes)
    fd = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return fd
    for i in range(period, n):
        window = closes[i - period + 1:i + 1]
        lengths = []
        for k in [1, 2, 4, 8]:
            if k >= period // 2:
                continue
            num_seg = (period - 1) // k
            if num_seg < 1:
                continue
            path = 0.0
            for m in range(k):
                idx_arr = list(range(m, period, k))
                if len(idx_arr) < 2:
                    continue
                for j in range(1, len(idx_arr)):
                    path += abs(window[idx_arr[j]] - window[idx_arr[j - 1]])
            nl = path * (period - 1) / (k * k * num_seg) if num_seg > 0 else 0
            if nl > 0:
                lengths.append((np.log(1.0 / k), np.log(nl)))
        if len(lengths) >= 2:
            x = np.array([l[0] for l in lengths])
            y = np.array([l[1] for l in lengths])
            fd[i] = np.clip(np.polyfit(x, y, 1)[0], 1.0, 2.0)
        else:
            fd[i] = 1.5
    return fd


class StrategyV19(TimeSeriesStrategy):
    """
    策略简介：梯度提升信号 + 分形维度过滤的趋势策略。

    使用指标：
    - Boosting Signal(60): 多特征集成信号
    - Fractal Dimension(30): 仅在趋势环境(FD<1.4)中使用boosting信号
    - RSI, ROC, EMA距离, 回归斜率: 输入特征
    - ATR(14): 止损距离计算

    进场条件（做多）：FD<1.4(趋势), Boosting信号>0.3
    进场条件（做空）：FD<1.4(趋势), Boosting信号<-0.3

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - FD上升到>1.6(转为震荡)或信号反转

    优点：FD过滤震荡市中的boosting假信号，只在趋势环境中交易
    缺点：双重指标延迟叠加，进场信号可能偏慢
    """
    name = "ag_alltime_v19"
    warmup = 300
    freq = "4h"

    boost_lookback: int = 60      # Optuna: 30-120
    fd_period: int = 30           # Optuna: 20-60
    trend_fd: float = 1.4        # Optuna: 1.2-1.5
    signal_thresh: float = 0.3   # Optuna: 0.1-0.5
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._signal = None
        self._fd = None
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
        self._fd = _fractal_dimension(closes, period=self.fd_period)

        rsi_arr = rsi(closes, period=14)
        roc_arr = rate_of_change(closes, period=20)
        ema_arr = ema(closes, period=50)
        ema_dist = np.full_like(closes, np.nan)
        valid_ema = ~np.isnan(ema_arr) & (ema_arr > 0)
        ema_dist[valid_ema] = (closes[valid_ema] - ema_arr[valid_ema]) / ema_arr[valid_ema] * 100
        slope_arr = linear_regression_slope(closes, period=20)
        features = [rsi_arr, roc_arr, ema_dist, slope_arr]
        self._signal = _boosting_signal(features, closes, lookback=self.boost_lookback)

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
        sig = self._signal[i]
        fd = self._fd[i]
        if np.isnan(sig) or np.isnan(fd):
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

        # 3. Signal-based exit: FD becomes choppy or signal reverses
        if side == 1 and (fd > 1.6 or sig < -0.1):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (fd > 1.6 or sig > 0.1):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry: FD confirms trending + boosting signal
        if side == 0:
            if fd < self.trend_fd and sig > self.signal_thresh:
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
            elif fd < self.trend_fd and sig < -self.signal_thresh:
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
            if self.direction == 1 and fd < self.trend_fd and sig > self.signal_thresh:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and fd < self.trend_fd and sig < -self.signal_thresh:
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
