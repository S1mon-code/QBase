import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.kmeans_regime import kmeans_regime
from indicators.trend.adx import adx
from indicators.momentum.macd import macd
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV142(TimeSeriesStrategy):
    """
    策略简介：4h K-Means regime + 1h ADX趋势 + 10min MACD入场的三周期策略。

    使用指标：
    - K-Means Regime(3 clusters) [4h]: 聚类识别趋势/震荡/下跌状态
    - ADX(14) [1h]: 中周期趋势强度确认
    - MACD(12,26,9) [10min]: 小周期动量交叉入场
    - ATR(14) [10min]: 止损距离

    进场条件（做多）：4h bull regime + 1h ADX>25 + 10min MACD hist从负转正
    出场条件：ATR追踪止损, 分层止盈, regime切换

    优点：ML聚类自适应识别市场状态，三重验证
    缺点：K-Means聚类结果可能不稳定
    """
    name = "medium_trend_v142"
    freq = "10min"
    warmup = 3000

    adx_threshold: float = 25.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._macd_hist = None
        self._atr = None
        self._avg_volume = None
        self._regime_4h = None
        self._bull_state = None
        self._adx_1h = None
        self._4h_map = None
        self._1h_map = None

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
        n = len(closes)

        # 10min indicators
        _, _, hist = macd(closes, fast=12, slow=26, signal=9)
        self._macd_hist = hist
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        # 1h from 10min (step=6)
        step_1h = 6
        n_1h = n // step_1h
        trim_1h = n_1h * step_1h
        closes_1h = closes[:trim_1h].reshape(n_1h, step_1h)[:, -1]
        highs_1h = highs[:trim_1h].reshape(n_1h, step_1h).max(axis=1)
        lows_1h = lows[:trim_1h].reshape(n_1h, step_1h).min(axis=1)
        self._adx_1h = adx(highs_1h, lows_1h, closes_1h, period=14)
        self._1h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step_1h - 1),
                                  n_1h - 1)

        # 4h from 10min (step=24)
        step_4h = 24
        n_4h = n // step_4h
        trim_4h = n_4h * step_4h
        closes_4h = closes[:trim_4h].reshape(n_4h, step_4h)[:, -1]

        # Build features for K-Means
        returns_4h = np.zeros_like(closes_4h)
        returns_4h[1:] = np.diff(closes_4h) / closes_4h[:-1]
        vol_4h = np.zeros_like(closes_4h)
        for k in range(20, len(closes_4h)):
            vol_4h[k] = np.std(returns_4h[k-20:k])
        features = np.column_stack([returns_4h, vol_4h])

        self._regime_4h = kmeans_regime(features, period=120, n_clusters=3)
        # Identify bull state
        state_means = {}
        for s in range(3):
            mask = self._regime_4h[1:] == s
            if mask.sum() > 0:
                state_means[s] = np.nanmean(returns_4h[1:][mask])
            else:
                state_means[s] = -999.0
        self._bull_state = max(state_means, key=state_means.get)
        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step_4h - 1),
                                  n_4h - 1)

    def on_bar(self, context):
        i = context.bar_index
        j4 = self._4h_map[i]
        j1 = self._1h_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        mh = self._macd_hist[i]
        atr_val = self._atr[i]
        regime = self._regime_4h[j4]
        adx_val = self._adx_1h[j1]
        if np.isnan(mh) or np.isnan(atr_val) or np.isnan(regime) or np.isnan(adx_val):
            return

        prev_mh = self._macd_hist[i - 1] if i > 0 else np.nan
        is_bull = (regime == self._bull_state)
        trending = (adx_val > self.adx_threshold)
        macd_cross = (not np.isnan(prev_mh) and prev_mh <= 0 and mh > 0)
        self.bars_since_last_scale += 1

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

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

        if side == 1 and not is_bull:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and is_bull and trending and macd_cross:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and is_bull and trending and mh > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
