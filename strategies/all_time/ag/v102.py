import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.ema_ribbon import ema_ribbon, ema_ribbon_signal
from indicators.microstructure.amihud import amihud_illiquidity
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class EMARibbonAmihudStrategy(TimeSeriesStrategy):
    """
    策略简介：EMA Ribbon趋势扩展 + Amihud流动性过滤的多空策略。

    使用指标：
    - EMA Ribbon Signal(8,13,21,34,55): 多条EMA排列方向，判断趋势强度
    - Amihud Illiquidity(20): 流动性冲击比率，过滤低流动性环境
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - EMA Ribbon signal = +1（所有EMA多头排列）
    - Amihud z-score < 1.5（流动性正常，非极端低流动性）

    进场条件（做空）：
    - EMA Ribbon signal = -1（所有EMA空头排列）
    - Amihud z-score < 1.5（流动性正常）

    出场条件：
    - ATR追踪止损 / 分层止盈 / Ribbon signal变为0或反转 / 流动性恶化

    优点：Ribbon排列确认趋势强度，Amihud过滤滑点风险大的行情
    缺点：Ribbon排列建立需要时间，错过趋势初期
    """
    name = "v102_ema_ribbon_amihud"
    warmup = 300
    freq = "4h"

    ribbon_periods: tuple = (8, 13, 21, 34, 55)
    amihud_period: int = 20         # Optuna: 10-40
    amihud_threshold: float = 1.5   # Optuna: 1.0-3.0
    atr_stop_mult: float = 3.0     # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._ribbon_signal = None
        self._amihud_zscore = None
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

        self._ribbon_signal = ema_ribbon_signal(closes, self.ribbon_periods)
        illiq, self._amihud_zscore = amihud_illiquidity(
            closes, volumes, period=self.amihud_period)
        self._atr = atr(highs, lows, closes, period=14)

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
        ribbon_sig = self._ribbon_signal[i]
        amihud_z = self._amihud_zscore[i]
        if np.isnan(ribbon_sig) or np.isnan(amihud_z):
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

        # 3. Signal-based exit: ribbon flip or liquidity crisis
        if side == 1 and (ribbon_sig <= 0 or amihud_z > 2.5):
            context.close_long()
            self._reset_state()
        elif side == -1 and (ribbon_sig >= 0 or amihud_z > 2.5):
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry
        if side == 0:
            liquidity_ok = amihud_z < self.amihud_threshold
            if ribbon_sig == 1 and liquidity_ok:
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
            elif ribbon_sig == -1 and liquidity_ok:
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
            if self.direction == 1 and ribbon_sig == 1 and amihud_z < self.amihud_threshold:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and ribbon_sig == -1 and amihud_z < self.amihud_threshold:
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
