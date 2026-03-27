import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.online_regression import online_sgd_signal
from indicators.ml.regime_persistence import regime_duration
from indicators.structure.smart_money import smart_money_index

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV122(TimeSeriesStrategy):
    """
    策略简介：LSTM-like Online SGD 方向信号 + 行情持续期过滤 + 智能资金确认的多空策略。

    使用指标：
    - Online SGD Signal(20): 自适应学习方向（模拟LSTM在线学习）
    - Regime Duration(60): 年轻行情更具动量，过滤老化趋势
    - Smart Money Index(20): 机构资金流向确认

    进场条件（做多）：SGD 信号>0，行情持续期<中位数（年轻行情），SMI 上升
    进场条件（做空）：SGD 信号<0，行情持续期<中位数，SMI 下降

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - SGD 信号反转

    优点：在线学习自适应市场变化，智能资金提供真实机构视角
    缺点：SGD 对噪音敏感，智能资金指标在低量时不可靠
    """
    name = "ag_alltime_v122"
    warmup = 400
    freq = "4h"

    sgd_lr: float = 0.01
    sgd_period: int = 20
    regime_period: int = 60
    duration_thresh: float = 30.0
    smi_period: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._sgd = None
        self._duration = None
        self._smi = None
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
        opens = context.get_full_open_array()
        volumes = context.get_full_volume_array()

        self._atr = atr(highs, lows, closes, period=14)

        # Build features for SGD
        from indicators.momentum.rsi import rsi
        from indicators.trend.adx import adx as adx_fn
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx_fn(highs, lows, closes, period=14)
        features = np.column_stack([rsi_arr, adx_arr, self._atr])
        self._sgd = online_sgd_signal(closes, features, learning_rate=self.sgd_lr, period=self.sgd_period)

        # Regime duration from a simple regime classification
        from indicators.regime.momentum_regime import momentum_regime
        regime_labels = momentum_regime(closes, fast=10, slow=60)
        self._duration = regime_duration(regime_labels, period=self.regime_period)

        self._smi = smart_money_index(opens, closes, highs, lows, volumes, period=self.smi_period)

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
        if np.isnan(atr_val) or atr_val <= 0:
            return
        sgd_val = self._sgd[i]
        dur_val = self._duration[i]
        smi_val = self._smi[i]
        if np.isnan(sgd_val) or np.isnan(dur_val) or np.isnan(smi_val):
            return

        # SMI direction: compare with previous
        smi_prev = self._smi[i - 1] if i > 0 else np.nan
        if np.isnan(smi_prev):
            return
        smi_rising = smi_val > smi_prev
        smi_falling = smi_val < smi_prev

        young_regime = dur_val < self.duration_thresh

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

        # 3. Signal-based exit
        if side == 1 and sgd_val < 0:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and sgd_val > 0:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and young_regime:
            if sgd_val > 0 and smi_rising:
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
            elif sgd_val < 0 and smi_falling:
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
            if self.direction == 1 and sgd_val > 0:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and sgd_val < 0:
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
