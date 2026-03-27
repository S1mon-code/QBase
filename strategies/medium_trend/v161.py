import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.volatility.atr import atr
from indicators.ml.ensemble_signal import ensemble_vote
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV161(TimeSeriesStrategy):
    """
    策略简介：三模型集成投票 + 成交量确认的ML做多趋势策略（日线）。

    使用指标：
    - Ensemble Vote (3 models, period=120): ML集成信号，Ridge/Lasso/RF投票
    - ADX(14): 趋势强度过滤，确认趋势存在
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Ensemble vote_score > 0.3（多数模型看涨）
    - Ensemble agreement >= 0.67（至少2/3模型一致）
    - ADX > adx_threshold（趋势足够强）

    出场条件：
    - ATR追踪止损（最高价回撤 N×ATR）
    - 分层止盈（3ATR / 5ATR）
    - vote_score < -0.3（模型共识转空）

    优点：多模型投票降低单一模型过拟合风险，agreement过滤低置信度信号
    缺点：ML模型需要足够warmup数据，震荡市投票可能频繁翻转
    """
    name = "mt_v161"
    warmup = 250
    freq = "daily"

    ml_period: int = 120
    adx_threshold: float = 20.0    # Optuna: 15-35
    vote_threshold: float = 0.3    # Optuna: 0.1-0.5
    atr_stop_mult: float = 3.0     # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._vote = None
        self._agreement = None
        self._adx = None
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
        self._adx = adx(highs, lows, closes, period=14)

        rsi_arr = rsi(closes, period=14)
        adx_arr = self._adx.copy()
        atr_arr = self._atr.copy()
        features = np.column_stack([rsi_arr, adx_arr, atr_arr])
        self._vote, self._agreement = ensemble_vote(closes, features, period=self.ml_period)
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
        vote_val = self._vote[i]
        agree_val = self._agreement[i]
        adx_val = self._adx[i]
        if np.isnan(vote_val) or np.isnan(agree_val) or np.isnan(adx_val):
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

        # 3. Signal exit: models turn bearish
        if side == 1 and vote_val < -0.3:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and vote_val > self.vote_threshold and agree_val >= 0.67 and adx_val > self.adx_threshold:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, vote_val, adx_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, vote_val, adx_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if vote_val < 0.1 or adx_val < self.adx_threshold:
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
