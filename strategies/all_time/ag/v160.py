import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.nearest_neighbor_signal import knn_signal
from indicators.trend.ema import ema_cross
from indicators.structure.smart_money import smart_money_index

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV160(TimeSeriesStrategy):
    """
    策略简介：Neural Net模拟(KNN信号) + EMA交叉 + Smart Money指数的多空策略。

    使用指标：
    - Nearest Neighbor Signal(60): KNN近邻预测方向，模拟简单NN
    - EMA Cross(12, 26): 快慢均线交叉信号
    - Smart Money Index(20): 机构资金流向，SMI上升=机构看多
    - ATR(14): 止损距离计算

    进场条件（做多）：KNN信号>0.5 + EMA金叉 + SMI上升
    进场条件（做空）：KNN信号<0.5 + EMA死叉 + SMI下降

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - EMA交叉反转

    优点：KNN非参数模型自适应，SMI捕捉机构行为
    缺点：KNN对特征尺度敏感，SMI在日内数据近似度低
    """
    name = "ag_alltime_v160"
    warmup = 400
    freq = "4h"

    ema_fast: int = 12             # Optuna: 8-20
    ema_slow: int = 26             # Optuna: 20-50
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._avg_volume = None
        self._knn_signal = None
        self._ema_signal = None
        self._fast_ema = None
        self._slow_ema = None
        self._smi = None
        self._smi_signal = None

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
        self._fast_ema, self._slow_ema, self._ema_signal = ema_cross(
            closes, fast_period=self.ema_fast, slow_period=self.ema_slow
        )

        self._smi, self._smi_signal = smart_money_index(
            opens, closes, highs, lows, volumes, period=20
        )

        # Build features for KNN
        from indicators.momentum.rsi import rsi
        rsi_arr = rsi(closes, period=14)
        features = np.column_stack([rsi_arr, self._atr])

        self._knn_signal, _ = knn_signal(closes, features, period=60)

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

        knn_pred = self._knn_signal[i]
        ema_sig = self._ema_signal[i]
        smi_val = self._smi[i]
        smi_sig = self._smi_signal[i]
        if np.isnan(knn_pred) or np.isnan(ema_sig) or np.isnan(smi_val) or np.isnan(smi_sig):
            return

        smi_rising = smi_val > smi_sig
        smi_falling = smi_val < smi_sig

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

        # 3. Signal exit: EMA cross reversal
        if side == 1 and ema_sig == -1:
            context.close_long()
            self._reset_state()
        elif side == -1 and ema_sig == 1:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: KNN direction + EMA cross + Smart Money
        if side == 0:
            if knn_pred > 0 and ema_sig == 1 and smi_rising:
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
            elif knn_pred < 0 and ema_sig == -1 and smi_falling:
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
        elif side != 0 and self._should_add(price, atr_val, ema_sig):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if self.direction == 1:
                    context.buy(add_lots)
                else:
                    context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, ema_sig):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1:
            if price < self.entry_price + atr_val:
                return False
            if ema_sig == -1:
                return False
        elif self.direction == -1:
            if price > self.entry_price - atr_val:
                return False
            if ema_sig == 1:
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
