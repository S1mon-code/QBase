import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.kalman_adaptive import adaptive_kalman
from indicators.regime.momentum_regime import momentum_regime
from indicators.volume.mfi import mfi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV127(TimeSeriesStrategy):
    """
    策略简介：Gaussian Process-like Adaptive Kalman 预测 + 行情动量 + MFI 资金流的多空策略。

    使用指标：
    - Adaptive Kalman(60): GP 风格自适应滤波预测方向
    - Momentum Regime(10,60): 快慢动量判断行情状态
    - MFI(14): 资金流入/流出确认

    进场条件（做多）：Kalman 预测上行，动量行情=上涨，MFI > 50
    进场条件（做空）：Kalman 预测下行，动量行情=下跌，MFI < 50

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Kalman 方向反转

    优点：Adaptive Kalman 自动调节噪音滤波强度，MFI 验证资金真实性
    缺点：Kalman 滤波器对初始参数敏感，MFI 在低量市场不准
    """
    name = "ag_alltime_v127"
    warmup = 400
    freq = "daily"

    kalman_period: int = 60
    mom_fast: int = 10
    mom_slow: int = 60
    mfi_period: int = 14
    mfi_thresh: float = 50.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._kalman = None
        self._mom_regime = None
        self._mfi = None
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

        self._atr = atr(highs, lows, closes, period=14)
        self._kalman = adaptive_kalman(closes, period=self.kalman_period)
        self._mom_regime = momentum_regime(closes, fast=self.mom_fast, slow=self.mom_slow)
        self._mfi = mfi(highs, lows, closes, volumes, period=self.mfi_period)

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
        kalman_val = self._kalman[i]
        kalman_prev = self._kalman[i - 1] if i > 0 else np.nan
        mom_val = self._mom_regime[i]
        mfi_val = self._mfi[i]
        if np.isnan(kalman_val) or np.isnan(kalman_prev) or np.isnan(mom_val) or np.isnan(mfi_val):
            return

        kalman_up = kalman_val > kalman_prev
        kalman_down = kalman_val < kalman_prev
        # Momentum regime: 1=bullish, -1=bearish, 0=neutral
        mom_bull = mom_val > 0
        mom_bear = mom_val < 0

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
        if side == 1 and kalman_down:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and kalman_up:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if kalman_up and mom_bull and mfi_val > self.mfi_thresh:
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
            elif kalman_down and mom_bear and mfi_val < self.mfi_thresh:
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
            if self.direction == 1 and kalman_up:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and kalman_down:
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
