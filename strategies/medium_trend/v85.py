import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.regime.fractal_dimension import fractal_dim
from indicators.momentum.stochastic import stochastic
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV85(TimeSeriesStrategy):
    """
    策略简介：Fractal Dimension趋势/震荡判断 + Stochastic超卖入场的4h多头策略。

    使用指标：
    - Fractal Dimension(60): < 1.5时市场处于趋势状态，作为过滤器
    - Stochastic(14,3): %K从超卖区回升作为入场信号
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Fractal Dim < 1.5（市场处于趋势状态）
    - Stochastic %K从<20回升至>20（超卖回升）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - Stochastic %K > 80（超买区信号弱化）

    优点：分形维度有效区分趋势/震荡，随机指标精准入场
    缺点：分形维度计算窗口大，信号滞后
    """
    name = "medium_trend_v85"
    warmup = 300
    freq = "4h"

    fd_period: int = 60           # Optuna: 40-100
    fd_threshold: float = 1.5     # Optuna: 1.3-1.6
    stoch_k: int = 14             # Optuna: 9-21
    stoch_d: int = 3
    stoch_oversold: float = 20.0  # Optuna: 15-30
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._fd = None
        self._stoch_k = None
        self._stoch_d = None
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

        self._fd = fractal_dim(closes, period=self.fd_period)
        self._stoch_k, self._stoch_d = stochastic(
            highs, lows, closes, k=self.stoch_k, d=self.stoch_d
        )
        self._atr = atr(highs, lows, closes, period=14)
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
        fd_val = self._fd[i]
        sk = self._stoch_k[i]
        if np.isnan(fd_val) or np.isnan(sk):
            return
        prev_sk = self._stoch_k[i - 1] if i > 0 else np.nan
        if np.isnan(prev_sk):
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

        # 3. Signal exit: overbought
        if side == 1 and sk > 80:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry: trending regime + stochastic oversold bounce
        is_trending = fd_val < self.fd_threshold
        oversold_bounce = prev_sk < self.stoch_oversold and sk >= self.stoch_oversold
        if side == 0 and is_trending and oversold_bounce:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, fd_val, sk):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, fd_val, sk):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if fd_val >= self.fd_threshold:
            return False
        if sk > 70:
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
