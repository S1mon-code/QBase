import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.pca_features import rolling_pca
from indicators.trend.ichimoku import ichimoku
from indicators.spread.gold_silver_ratio import gold_silver_ratio
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx
from indicators.momentum.roc import rate_of_change

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV199(TimeSeriesStrategy):
    """
    策略简介：PCA降维方向信号 + Ichimoku Cloud位置 + 金银比的ML跨品种策略。

    使用指标：
    - Rolling PCA(60, 3): 将RSI/ADX/ATR/ROC压缩为主成分，PC1方向为综合趋势
    - Ichimoku(9,26,52,26): 云图位置确认，价格在云上/下方
    - Gold/Silver Ratio(60): 金银比z-score，确认白银相对估值
    - ATR(14): 止损距离计算

    进场条件（做多）：PC1>0（综合上涨）+ 价格在云上方 + 金银比z>0
    进场条件（做空）：PC1<0（综合下跌）+ 价格在云下方 + 金银比z<0

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - PC1方向反转 或 价格进入云内

    优点：PCA综合多维信号避免过度拟合单一指标，跨品种视角
    缺点：PCA方向解释性较弱，重训练成本高
    """
    name = "ag_alltime_v199"
    warmup = 200
    freq = "daily"

    pca_period: int = 60          # Optuna: 40-80
    pca_components: int = 3       # Optuna: 2-4
    ichi_tenkan: int = 9          # Optuna: 7-12
    ichi_kijun: int = 26          # Optuna: 20-34
    gsr_period: int = 60          # Optuna: 40-80
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._pca_comp = None
        self._tenkan = None
        self._kijun = None
        self._senkou_a = None
        self._senkou_b_line = None
        self._gsr_ratio = None
        self._gsr_zscore = None
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
        au_closes = context.load_auxiliary_close("AU")

        # Build features for PCA
        rsi_arr = rsi(closes, 14)
        adx_arr = adx(highs, lows, closes, 14)
        atr_arr = atr(highs, lows, closes, 14)
        roc_arr = rate_of_change(closes, 12)
        self._atr = atr_arr

        features = np.column_stack([rsi_arr, adx_arr, atr_arr, roc_arr])
        self._pca_comp, _ = rolling_pca(
            features, period=self.pca_period, n_components=self.pca_components)

        disp = self.ichi_kijun
        self._tenkan, self._kijun, self._senkou_a, self._senkou_b_line, _ = ichimoku(
            highs, lows, closes, self.ichi_tenkan, self.ichi_kijun, 52, disp)
        self._gsr_ratio, self._gsr_zscore = gold_silver_ratio(
            au_closes, closes, self.gsr_period)

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

        # PCA first component as direction
        pc1 = self._pca_comp[i, 0] if i < len(self._pca_comp) else np.nan
        if np.isnan(pc1):
            return

        # Ichimoku cloud
        sa = self._senkou_a[i] if i < len(self._senkou_a) else np.nan
        sb = self._senkou_b_line[i] if i < len(self._senkou_b_line) else np.nan
        if np.isnan(sa) or np.isnan(sb):
            return

        cloud_top = max(sa, sb)
        cloud_bot = min(sa, sb)

        gsr_z = self._gsr_zscore[i] if i < len(self._gsr_zscore) else 0.0
        if np.isnan(gsr_z):
            gsr_z = 0.0

        prev_pc1 = self._pca_comp[i - 1, 0] if i > 0 and i - 1 < len(self._pca_comp) else np.nan
        if np.isnan(prev_pc1):
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

        # 3. Signal exit: PC1 reversal or price enters cloud
        in_cloud = cloud_bot <= price <= cloud_top
        if side == 1 and (pc1 < 0 or in_cloud):
            context.close_long()
            self._reset_state()
        elif side == -1 and (pc1 > 0 or in_cloud):
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: PC1 direction + cloud position + Au/Ag ratio
        if side == 0:
            if pc1 > 0 and price > cloud_top and gsr_z > 0:
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
            elif pc1 < 0 and price < cloud_bot and gsr_z < 0:
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
            if self.direction == 1 and pc1 > 0 and price > cloud_top:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and pc1 < 0 and price < cloud_bot:
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
