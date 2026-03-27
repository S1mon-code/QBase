import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.autoencoder_error import reconstruction_error
from indicators.ml.regime_persistence import regime_duration
from indicators.trend.supertrend import supertrend

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV165(TimeSeriesStrategy):
    """
    策略简介：Autoencoder异常检测 + Regime稳定性 + Supertrend方向的做多策略（4h）。

    使用指标：
    - Reconstruction Error: PCA-based异常检测，低error=正常市场状态
    - Regime Duration: 当前regime持续时间，长持续=稳定趋势
    - Supertrend(10,3): 趋势方向确认
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Supertrend direction = 1（上升趋势）
    - reconstruction_error < median（市场正常，非异常状态）
    - current_regime_duration > 10（regime稳定）

    出场条件：
    - ATR追踪止损 / 分层止盈
    - Supertrend翻转为-1
    - reconstruction_error突增（异常市场状态，提前退出）

    优点：异常检测过滤极端行情，regime持续性确认趋势稳定
    缺点：异常阈值难以确定，可能在趋势加速期误判为异常
    """
    name = "mt_v165"
    warmup = 400
    freq = "4h"

    st_period: int = 10             # Optuna: 7-20
    st_multiplier: float = 3.0      # Optuna: 2.0-5.0
    atr_stop_mult: float = 3.0      # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._st_dir = None
        self._recon_err = None
        self._regime_dur = None
        self._err_median = None
        self._atr = None
        self._avg_volume = None

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

        self._atr = atr(highs, lows, closes, period=14)
        _, self._st_dir = supertrend(highs, lows, closes, period=self.st_period, multiplier=self.st_multiplier)

        from indicators.momentum.rsi import rsi
        rsi_arr = rsi(closes, 14)
        features = np.column_stack([rsi_arr, self._atr, closes])
        self._recon_err = reconstruction_error(features, period=120, encoding_dim=2)

        from indicators.ml.kmeans_regime import kmeans_regime
        labels, _ = kmeans_regime(features, period=120, n_clusters=3)
        self._regime_dur, _, _ = regime_duration(labels, period=60)

        # Rolling median of reconstruction error for threshold
        n = len(closes)
        self._err_median = np.full(n, np.nan)
        window = 60
        for idx in range(window, n):
            seg = self._recon_err[idx - window:idx]
            valid = seg[~np.isnan(seg)]
            if len(valid) > 0:
                self._err_median[idx] = np.median(valid)

        self._avg_volume = fast_avg_volume(volumes, 20)

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
        st_dir = self._st_dir[i]
        err = self._recon_err[i]
        err_med = self._err_median[i]
        dur = self._regime_dur[i]
        if np.isnan(st_dir) or np.isnan(err) or np.isnan(err_med) or np.isnan(dur):
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

        # 2. Tiered profit-taking
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

        # 3. Signal exit: trend flip or anomaly spike
        if side == 1 and (st_dir == -1 or err > err_med * 3):
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and st_dir == 1 and err < err_med and dur > 10:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, st_dir, err, err_med):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, st_dir, err, err_med):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if st_dir != 1 or err > err_med:
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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
