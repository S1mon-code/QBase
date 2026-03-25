import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.ensemble_signal import ensemble_vote
from indicators.ml.regime_persistence import regime_duration
from indicators.ml.kmeans_regime import kmeans_regime
from indicators.seasonality.monthly_pattern import monthly_seasonal
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class RFRegimeDurationMonthlyPatternStrategy(TimeSeriesStrategy):
    """
    策略简介：Ensemble ML（含 RF）在新 regime 开始时 + 有利月份效应确认的日频多空策略。

    使用指标：
    - Ensemble Vote(Ridge+Lasso+RF, 120): 三模型投票方向预测
    - Regime Duration(from K-Means, 60): 检测 regime 年龄，新 regime（duration < 10）=机会
    - Monthly Seasonal(3yr lookback): 月度季节性评分，正值=历史上有利月份
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Ensemble vote > 0.3（多数模型看多）
    - 当前 regime duration < 10（新 regime 刚开始）
    - Monthly seasonal > 0（历史有利月份）

    进场条件（做空）：
    - Ensemble vote < -0.3
    - 当前 regime duration < 10
    - Monthly seasonal < 0

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Ensemble vote 反转

    优点：多模型集成提高稳健性 + 新 regime 入场捕捉趋势起点 + 季节性增强胜率
    缺点：RF 过拟合风险，月效应在白银上不一定稳定
    """
    name = "v143_rf_regime_duration_monthly_pattern"
    warmup = 250
    freq = "daily"

    ens_period: int = 120
    vote_threshold: float = 0.3
    duration_threshold: int = 10
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._vote = None
        self._confidence = None
        self._regime_dur = None
        self._seasonal = None
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

        # Build features for ensemble and K-Means
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx(highs, lows, closes, period=14)
        atr_arr = atr(highs, lows, closes, period=14)
        features = np.column_stack([rsi_arr, adx_arr, atr_arr])

        self._vote, self._confidence = ensemble_vote(closes, features, period=self.ens_period)

        # K-Means for regime labels, then compute duration
        km_labels, _ = kmeans_regime(features, period=self.ens_period, n_clusters=3)
        self._regime_dur, _, _ = regime_duration(km_labels, period=60)

        # Monthly seasonal pattern
        datetimes = context.get_full_datetime_array()
        self._seasonal = monthly_seasonal(closes, datetimes, lookback_years=3)

        self._atr = atr_arr

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
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        vote_val = self._vote[i]
        dur_val = self._regime_dur[i]
        seasonal_val = self._seasonal[i]
        atr_val = self._atr[i]
        if np.isnan(vote_val) or np.isnan(atr_val):
            return
        if np.isnan(dur_val):
            dur_val = 999.0
        if np.isnan(seasonal_val):
            seasonal_val = 0.0

        self.bars_since_last_scale += 1
        is_fresh_regime = dur_val < self.duration_threshold

        # ── 1. 止损检查 ──
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
        if side == 1 and vote_val < -self.vote_threshold:
            context.close_long()
            self._reset_state()
        elif side == -1 and vote_val > self.vote_threshold:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if vote_val > self.vote_threshold and is_fresh_regime and seasonal_val > 0:
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
            elif vote_val < -self.vote_threshold and is_fresh_regime and seasonal_val < 0:
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

        # ── 5. 加仓逻辑 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and vote_val > self.vote_threshold):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and vote_val < -self.vote_threshold):
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
        self.direction = 0
