import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.kmeans_regime import kmeans_regime
from indicators.structure.smart_money import smart_money_index
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class KMeansSmartMoneyStrategy(TimeSeriesStrategy):
    """
    策略简介：K-Means聚类识别市场状态 + Smart Money指标确认机构方向。

    使用指标：
    - K-Means Regime(120, 3): 将市场状态聚为3类，识别趋势/震荡状态
    - Smart Money Index: 机构资金流向，SMI上升=机构买入
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - K-Means当前cluster的均值回报 > 0（聚类方向向上）
    - SMI > SMI signal line（机构资金流入）

    进场条件（做空）：
    - K-Means当前cluster的均值回报 < 0（聚类方向向下）
    - SMI < SMI signal line（机构资金流出）

    出场条件：
    - ATR追踪止损 / 分层止盈 / 信号反转

    优点：聚类自动适应市场状态变化，Smart Money过滤散户噪音
    缺点：K-Means非确定性，重新训练可能标签漂移
    """
    name = "v42_kmeans_smart_money"
    warmup = 400
    freq = "daily"

    kmeans_period: int = 120
    n_clusters: int = 3
    smi_period: int = 20
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._cluster_labels = None
        self._smi = None
        self._smi_signal = None
        self._atr = None
        self._avg_volume = None
        self._cluster_returns = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        opens = context.get_full_open_array()
        volumes = context.get_full_volume_array()
        n = len(closes)

        rsi_arr = rsi(closes, 14)
        atr_arr = atr(highs, lows, closes, 14)
        # Normalize for clustering
        returns = np.full(n, np.nan)
        returns[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
        features = np.column_stack([rsi_arr, atr_arr / np.maximum(closes, 1e-9), returns])

        self._cluster_labels, _ = kmeans_regime(features, period=self.kmeans_period,
                                                 n_clusters=self.n_clusters)

        # Pre-compute cluster mean returns
        self._cluster_returns = np.full(n, np.nan)
        lookback = 60
        for idx in range(lookback, n):
            label = self._cluster_labels[idx]
            if np.isnan(label):
                continue
            label = int(label)
            mask = self._cluster_labels[idx - lookback:idx] == label
            rets_window = returns[idx - lookback:idx]
            valid = mask & ~np.isnan(rets_window)
            if np.sum(valid) > 5:
                self._cluster_returns[idx] = np.mean(rets_window[valid])

        self._smi, self._smi_signal = smart_money_index(
            opens, closes, highs, lows, volumes, period=self.smi_period)

        self._atr = atr(highs, lows, closes, period=self.atr_period)

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        cluster_ret = self._cluster_returns[i]
        smi_val = self._smi[i]
        smi_sig = self._smi_signal[i]
        atr_val = self._atr[i]
        if np.isnan(cluster_ret) or np.isnan(smi_val) or np.isnan(smi_sig) or np.isnan(atr_val) or atr_val <= 0:
            return

        self.bars_since_last_scale += 1
        bull_cluster = cluster_ret > 0
        bear_cluster = cluster_ret < 0
        smi_bull = smi_val > smi_sig
        smi_bear = smi_val < smi_sig

        # ── 1. 止损 ──
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

        # ── 2. 分层止盈 ──
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
        elif side == -1 and self.entry_price > 0:
            profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_3atr = True
                return

        # ── 3. 信号反转退出 ──
        if side == 1 and (bear_cluster or smi_bear):
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and (bull_cluster or smi_bull):
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场 ──
        if side == 0:
            if bull_cluster and smi_bull:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
            elif bear_cluster and smi_bear:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0

        # ── 5. 加仓 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and bull_cluster and smi_bull):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and bear_cluster and smi_bear):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.sell(add)
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
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
