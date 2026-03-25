import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.spectral_clustering_regime import spectral_regime
from indicators.regime.volatility_clustering import vol_clustering
from indicators.momentum.stochastic import stochastic

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV129(TimeSeriesStrategy):
    """
    策略简介：Spectral 聚类周期 + 羊群行为（波动率聚集）+ Stochastic 确认的多空策略。

    使用指标：
    - Spectral Regime(120): 频谱聚类识别市场周期状态
    - Vol Clustering(60): 波动率聚集度作为羊群行为代理
    - Stochastic(14,3): 超买超卖动量确认

    进场条件（做多）：Spectral 状态=上涨，波动率聚集高（羊群跟风），%K>%D且<80
    进场条件（做空）：Spectral 状态=下跌，波动率聚集高，%K<%D且>20

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Spectral 状态转换

    优点：频谱分析捕捉隐含周期，羊群行为放大趋势信号
    缺点：聚类结果不稳定，标签可能在不同窗口间跳跃
    """
    name = "ag_alltime_v129"
    warmup = 400
    freq = "daily"

    spectral_period: int = 120
    vol_clust_period: int = 60
    vol_clust_thresh: float = 0.5
    stoch_k: int = 14
    stoch_d: int = 3
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._spectral = None
        self._vol_clust = None
        self._stoch_k = None
        self._stoch_d = None
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

        # Spectral clustering needs features
        from indicators.momentum.rsi import rsi
        rsi_arr = rsi(closes, period=14)
        features = np.column_stack([rsi_arr, self._atr, closes])
        self._spectral = spectral_regime(features, period=self.spectral_period, n_clusters=3)

        self._vol_clust = vol_clustering(closes, period=self.vol_clust_period)

        k_arr, d_arr = stochastic(highs, lows, closes, k=self.stoch_k, d=self.stoch_d)
        self._stoch_k = k_arr
        self._stoch_d = d_arr

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
        spec_val = self._spectral[i]
        vc_val = self._vol_clust[i]
        sk_val = self._stoch_k[i]
        sd_val = self._stoch_d[i]
        if np.isnan(spec_val) or np.isnan(vc_val) or np.isnan(sk_val) or np.isnan(sd_val):
            return

        # Determine spectral direction from price context
        # Regime labels are 0,1,2 - map by comparing with recent return
        closes_arr = context.get_full_close_array()
        if i < 5:
            return
        recent_ret = (closes_arr[i] - closes_arr[i - 5]) / closes_arr[i - 5]
        spec_bull = recent_ret > 0 and int(spec_val) != 0  # not in neutral regime
        spec_bear = recent_ret < 0 and int(spec_val) != 0

        herding = vc_val > self.vol_clust_thresh

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

        # 3. Signal-based exit: stochastic reversal
        if side == 1 and sk_val > 80 and sk_val < sd_val:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and sk_val < 20 and sk_val > sd_val:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and herding:
            if spec_bull and sk_val > sd_val and sk_val < 80:
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
            elif spec_bear and sk_val < sd_val and sk_val > 20:
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
            if self.direction == 1 and sk_val > sd_val:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and sk_val < sd_val:
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
