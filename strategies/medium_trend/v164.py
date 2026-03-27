import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.online_regression import online_sgd_signal
from indicators.regime.regime_switch_speed import switch_speed
from indicators.trend.hma import hma

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV164(TimeSeriesStrategy):
    """
    策略简介：在线学习SGD + Regime切换速度 + HMA方向的做多策略（1h）。

    使用指标：
    - Online SGD Signal: 在线线性模型实时更新权重，预测方向
    - Regime Switch Speed: 行情切换频率，低切换=稳定趋势
    - HMA(20): Hull均线方向确认趋势
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - SGD signal > 0（模型预测看涨）
    - switch_frequency < 0.15（行情稳定，非频繁切换）
    - HMA上升（当前HMA > 前一根HMA）

    出场条件：
    - ATR追踪止损 / 分层止盈
    - SGD signal < 0（模型预测反转）

    优点：在线学习实时适应市场变化，HMA低延迟趋势确认
    缺点：SGD对学习率敏感，高波动时可能过度反应
    """
    name = "mt_v164"
    warmup = 800
    freq = "1h"

    sgd_lr: float = 0.01            # Optuna: 0.001-0.05
    hma_period: int = 20            # Optuna: 10-40
    atr_stop_mult: float = 3.0      # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._sgd_signal = None
        self._switch_freq = None
        self._hma = None
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
        self._hma = hma(closes, period=self.hma_period)

        from indicators.momentum.rsi import rsi
        from indicators.momentum.roc import rate_of_change
        rsi_arr = rsi(closes, 14)
        roc_arr = rate_of_change(closes, 12)
        features = np.column_stack([rsi_arr, roc_arr, self._atr])
        self._sgd_signal, _ = online_sgd_signal(closes, features, learning_rate=self.sgd_lr)

        from indicators.regime.vol_regime_markov import vol_regime_simple
        regime, _, _ = vol_regime_simple(closes, period=60)
        _, self._switch_freq, _ = switch_speed(regime, period=60)

        self._avg_volume = fast_avg_volume(volumes, 20)

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
        sgd = self._sgd_signal[i]
        sf = self._switch_freq[i]
        hma_now = self._hma[i]
        hma_prev = self._hma[i - 1] if i > 0 else np.nan
        if np.isnan(sgd) or np.isnan(sf) or np.isnan(hma_now) or np.isnan(hma_prev):
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

        # 3. Signal exit
        if side == 1 and sgd < 0:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and sgd > 0 and sf < 0.15 and hma_now > hma_prev:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, sgd):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, sgd):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if sgd <= 0:
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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
