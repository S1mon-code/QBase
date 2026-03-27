import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.kmeans_regime import kmeans_regime
from indicators.volatility.bollinger import bollinger_bands
from indicators.structure.smart_money import smart_money_index
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV197(TimeSeriesStrategy):
    """
    策略简介：K-Means聚类趋势regime + BB突破 + Smart Money方向的ML突破策略。

    使用指标：
    - K-Means Regime(120, 3): 基于RSI/ADX/ATR特征的聚类，识别趋势cluster
    - Bollinger Bands(20, 2): 价格突破上/下轨信号
    - Smart Money Index(20): 机构资金方向确认
    - ATR(14): 止损距离计算

    进场条件（做多）：K-Means在趋势cluster + 价格突破BB上轨 + SMI上升
    进场条件（做空）：K-Means在趋势cluster + 价格跌破BB下轨 + SMI下降

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - 价格回到BB中轨 或 regime切换

    优点：聚类自动识别适合突破的市场状态，减少震荡市假突破
    缺点：聚类标签不稳定可能导致信号抖动
    """
    name = "ag_alltime_v197"
    warmup = 200
    freq = "daily"

    kmeans_period: int = 120      # Optuna: 80-160
    kmeans_clusters: int = 3      # Optuna: 2-4
    bb_period: int = 20           # Optuna: 14-30
    bb_std: float = 2.0           # Optuna: 1.5-2.5
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._km_labels = None
        self._km_dist = None
        self._bb_upper = None
        self._bb_mid = None
        self._bb_lower = None
        self._smi = None
        self._smi_sig = None
        self._atr = None
        self._avg_volume = None
        self._trending_cluster = None

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
        opens = context.get_full_open_array()
        volumes = context.get_full_volume_array()

        # Build features for K-Means
        rsi_arr = rsi(closes, 14)
        adx_arr = adx(highs, lows, closes, 14)
        atr_arr = atr(highs, lows, closes, 14)
        self._atr = atr_arr

        features = np.column_stack([rsi_arr, adx_arr, atr_arr])
        self._km_labels, self._km_dist = kmeans_regime(
            features, period=self.kmeans_period, n_clusters=self.kmeans_clusters)

        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            closes, self.bb_period, self.bb_std)
        self._smi, self._smi_sig = smart_money_index(
            opens, closes, highs, lows, volumes, period=20)

        # Identify trending cluster: highest average ADX
        n = len(closes)
        cluster_adx = {}
        for c in range(self.kmeans_clusters):
            mask = self._km_labels == c
            valid = mask & ~np.isnan(adx_arr)
            if np.any(valid):
                cluster_adx[c] = np.nanmean(adx_arr[valid])
            else:
                cluster_adx[c] = 0.0
        self._trending_cluster = max(cluster_adx, key=cluster_adx.get)

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
        label = self._km_labels[i]
        bb_u = self._bb_upper[i]
        bb_m = self._bb_mid[i]
        bb_l = self._bb_lower[i]
        smi_val = self._smi[i]

        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(label):
            return
        if np.isnan(bb_u) or np.isnan(bb_l) or np.isnan(bb_m):
            return
        if np.isnan(smi_val):
            return

        prev_smi = self._smi[i - 1] if i > 0 else np.nan
        if np.isnan(prev_smi):
            return

        label = int(label)
        smi_rising = smi_val > prev_smi
        smi_falling = smi_val < prev_smi
        is_trending = label == self._trending_cluster

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

        # 3. Signal exit: price returns to BB mid or regime change
        if side == 1 and (price < bb_m or not is_trending):
            context.close_long()
            self._reset_state()
        elif side == -1 and (price > bb_m or not is_trending):
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: trending cluster + BB breakout + smart money
        if side == 0 and is_trending:
            if price > bb_u and smi_rising:
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
            elif price < bb_l and smi_falling:
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
        elif side != 0 and self._should_add(price, atr_val) and is_trending:
            if self.direction == 1 and price > bb_u:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and price < bb_l:
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
