import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.pca_features import rolling_pca
from indicators.regime.vol_regime_markov import vol_regime_simple
from indicators.volatility.bollinger import bollinger_bands
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx
from indicators.volatility.ttm_squeeze import ttm_squeeze

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV119(TimeSeriesStrategy):
    """
    策略简介：PCA主成分方向 + BB Squeeze释放 + 波动率regime过渡捕捉。

    使用指标：
    - Rolling PCA(60, 3): 多特征降维，PC1方向=主成分趋势方向
    - Vol Regime Simple(60): 2态波动率regime（低vol/高vol）
    - TTM Squeeze(20): Bollinger Squeeze检测
    - Bollinger Bands(20, 2.0): %B位置参考
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Squeeze释放（低vol->高vol过渡）
    - PCA PC1 > 0（主成分趋势向上）
    - Vol regime刚从低vol切换到高vol（transition_prob高）
    进场条件（做空）：
    - Squeeze释放
    - PCA PC1 < 0
    - Vol regime过渡中

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - PCA方向反转或重新进入squeeze

    优点：PCA自适应提取主要信号维度，squeeze+regime过渡精准定时
    缺点：PCA方向解释需要谨慎，regime分类存在滞后
    """
    name = "ag_alltime_v119"
    warmup = 300
    freq = "daily"

    pca_period: int = 60          # Optuna: 40-120
    vol_period: int = 60          # Optuna: 30-90
    bb_period: int = 20           # Optuna: 15-30
    trans_thresh: float = 0.3     # Optuna: 0.15-0.50
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._pc1 = None
        self._vol_regime = None
        self._trans_prob = None
        self._squeeze_on = None
        self._momentum = None
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
        n = len(closes)

        self._atr = atr(highs, lows, closes, period=14)

        # Features for PCA
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx(highs, lows, closes, period=14)
        atr_arr = self._atr.copy()
        atr_norm = np.full(n, np.nan)
        valid = closes > 1e-9
        atr_norm[valid] = atr_arr[valid] / closes[valid]

        roc_arr = np.full(n, np.nan)
        for idx in range(10, n):
            if closes[idx - 10] > 1e-9:
                roc_arr[idx] = (closes[idx] / closes[idx - 10]) - 1.0

        features = np.column_stack([rsi_arr, adx_arr, atr_norm, roc_arr])
        components, _ = rolling_pca(features, period=self.pca_period, n_components=3)
        self._pc1 = components[:, 0]  # First principal component

        # Vol regime
        self._vol_regime, self._trans_prob, _ = vol_regime_simple(
            closes, period=self.vol_period
        )

        # TTM Squeeze
        self._squeeze_on, self._momentum = ttm_squeeze(
            highs, lows, closes,
            bb_period=self.bb_period, bb_mult=2.0,
            kc_period=self.bb_period, kc_mult=1.5,
        )

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
        pc1 = self._pc1[i]
        vol_r = self._vol_regime[i]
        trans_p = self._trans_prob[i]
        sq_on = self._squeeze_on[i] if i < len(self._squeeze_on) else True
        mom = self._momentum[i]
        if np.isnan(pc1) or np.isnan(vol_r) or np.isnan(trans_p) or np.isnan(mom):
            return

        # Detect squeeze release
        squeeze_release = False
        if i >= 1 and not np.isnan(self._squeeze_on[i - 1]):
            squeeze_release = self._squeeze_on[i - 1] and not sq_on

        # Vol regime transition: low->high vol
        vol_transitioning = trans_p > self.trans_thresh

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

        # 3. Signal-based exit: PCA reversal or re-squeeze
        if side == 1 and (pc1 < 0 or sq_on):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (pc1 > 0 or sq_on):
            context.close_short()
            self._reset_state()
            return

        # Re-read position
        side, lots = context.position

        # 4. Entry: squeeze release + PCA direction + vol transition
        if side == 0 and squeeze_release and vol_transitioning:
            if pc1 > 0 and mom > 0:
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
            elif pc1 < 0 and mom < 0:
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
            signal_ok = (self.direction == 1 and pc1 > 0 and mom > 0) or \
                        (self.direction == -1 and pc1 < 0 and mom < 0)
            if signal_ok:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    if self.direction == 1:
                        context.buy(add_lots)
                    else:
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
