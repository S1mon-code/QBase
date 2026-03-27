import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.ml.ensemble_signal import ensemble_vote
from indicators.volume.volume_spike import volume_spike
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV84(TimeSeriesStrategy):
    """
    策略简介：Ensemble ML投票信号 + Volume Spike确认的4h多头策略。

    使用指标：
    - Ensemble Vote(120): Ridge/Lasso/RF三模型投票方向
    - Volume Spike(20, 2.0): 成交量突增确认资金流入
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Ensemble投票分数 > 0.3（多数模型看多）
    - 近期有Volume Spike（成交量放大确认）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - Ensemble投票分数 < -0.1（模型转空）

    优点：多模型投票减少单模型偏差，量能确认增加可靠性
    缺点：ML模型需要足够训练窗口，小样本不稳定
    """
    name = "medium_trend_v84"
    warmup = 400
    freq = "4h"

    ensemble_period: int = 120      # Optuna: 80-200
    vote_threshold: float = 0.3     # Optuna: 0.1-0.6
    spike_lookback: int = 5         # Optuna: 3-10
    atr_stop_mult: float = 3.0     # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._vote_score = None
        self._vote_agree = None
        self._vol_spike = None
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

        # Build feature matrix for ensemble: RSI, ADX, ROC
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx(highs, lows, closes, period=14)
        n = len(closes)
        roc_arr = np.full(n, np.nan)
        roc_arr[10:] = (closes[10:] - closes[:-10]) / closes[:-10]
        features = np.column_stack([rsi_arr, adx_arr, roc_arr])

        self._vote_score, self._vote_agree = ensemble_vote(
            closes, features, period=self.ensemble_period
        )
        self._vol_spike = volume_spike(volumes, period=20, threshold=2.0)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        vote = self._vote_score[i]
        if np.isnan(vote):
            return

        # Check for recent volume spike in last N bars
        recent_spike = False
        lb = max(0, i - self.spike_lookback)
        for j in range(lb, i + 1):
            if not np.isnan(self._vol_spike[j]) and self._vol_spike[j] > 0:
                recent_spike = True
                break

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

        # 3. Signal exit: ensemble turns bearish
        if side == 1 and vote < -0.1:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and vote > self.vote_threshold and recent_spike:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, vote):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, vote):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if vote < self.vote_threshold:
            return False
        return True

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
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
