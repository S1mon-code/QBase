import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.kmeans_regime import kmeans_regime
from indicators.momentum.stochastic import stochastic
from indicators.volatility.atr import atr
from indicators.trend.adx import adx
from indicators.momentum.rsi import rsi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV5(TimeSeriesStrategy):
    """
    策略简介：K-Means聚类行情识别 + Stochastic回调入场的做多策略（5min频率）。

    使用指标：
    - K-Means Regime(3 clusters): 基于RSI/ADX/ATR特征聚类识别行情状态
    - Stochastic(14,3): %K/%D超卖回调入场
    - ATR(14): 止损距离计算

    进场条件（做多）：K-Means判定为趋势状态 且 Stochastic %K < 30（超卖回调）
    出场条件：ATR追踪止损 / 分层止盈 / 状态切换 或 Stochastic %K > 80

    优点：聚类捕捉多维度状态，Stochastic精确回调入场
    缺点：聚类标签不稳定，需要足够历史数据
    """
    name = "mt_v5"
    warmup = 2000
    freq = "5min"

    kmeans_period: int = 120
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_entry: float = 30.0
    stoch_exit: float = 80.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._regime = None
        self._stoch_k = None
        self._stoch_d = None
        self._atr = None
        self._avg_volume = None
        self._trend_state = None

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

        rsi_arr = rsi(closes, 14)
        adx_arr = adx(highs, lows, closes, 14)
        atr_arr = atr(highs, lows, closes, 14)
        features = np.column_stack([rsi_arr, adx_arr, atr_arr])

        self._regime = kmeans_regime(features, period=self.kmeans_period, n_clusters=3)
        sk, sd = stochastic(highs, lows, closes, k=self.stoch_k, d=self.stoch_d)
        self._stoch_k = sk
        self._stoch_d = sd
        self._atr = atr_arr
        self._avg_volume = fast_avg_volume(volumes, 20)

        # Identify trend state: cluster with highest mean ADX
        n = len(closes)
        state_adx = {}
        for s in range(3):
            mask = self._regime == s
            if np.any(mask):
                valid = adx_arr[mask]
                valid = valid[~np.isnan(valid)]
                if len(valid) > 0:
                    state_adx[s] = np.mean(valid)
        self._trend_state = max(state_adx, key=state_adx.get) if state_adx else 0

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        regime = self._regime[i]
        sk = self._stoch_k[i]
        if np.isnan(atr_val) or np.isnan(regime) or np.isnan(sk):
            return

        self.bars_since_last_scale += 1
        is_trend = int(regime) == self._trend_state

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

        # 3. Signal exit
        if side == 1 and (not is_trend or sk > self.stoch_exit):
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and is_trend and sk < self.stoch_entry:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, is_trend):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, is_trend):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if not is_trend:
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
