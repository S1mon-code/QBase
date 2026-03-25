import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.attention_score import attention_weights
from indicators.momentum.tsi import tsi
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV150(TimeSeriesStrategy):
    """
    策略简介：Attention注意力加权信号 + 低离散度(ADX确认) + TSI方向的多空策略。

    使用指标：
    - Attention Weights(60): 特征注意力加权信号，自适应特征选择
    - ADX(14): 低离散度过滤（ADX>25=趋势明确）
    - TSI(25,13,7): 方向确认，TSI>0做多、TSI<0做空
    - ATR(14): 止损距离计算

    进场条件（做多）：attention信号>0 + ADX>25 + TSI>0且TSI>signal
    进场条件（做空）：attention信号<0 + ADX>25 + TSI<0且TSI<signal

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - TSI方向反转

    优点：注意力机制自适应权重，TSI双平滑减噪
    缺点：特征矩阵构建需要多指标，计算量大
    """
    name = "ag_alltime_v150"
    warmup = 300
    freq = "4h"

    adx_threshold: float = 25.0    # Optuna: 18-35
    atr_stop_mult: float = 3.0     # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._avg_volume = None
        self._attn_signal = None
        self._adx = None
        self._tsi_line = None
        self._tsi_signal = None

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
        self._adx = adx(highs, lows, closes, period=14)
        self._tsi_line, self._tsi_signal = tsi(closes, long_period=25, short_period=13, signal_period=7)

        # Build features for attention
        rsi_arr = rsi(closes, period=14)
        adx_arr = self._adx
        atr_arr = self._atr
        features = np.column_stack([rsi_arr, adx_arr, atr_arr])

        # Target = forward returns (shifted back for no look-ahead)
        n = len(closes)
        target = np.full(n, np.nan)
        target[:-1] = np.diff(closes) / closes[:-1]
        # Shift target back by 1 to avoid look-ahead
        target_lagged = np.full(n, np.nan)
        target_lagged[1:] = target[:-1]

        _, self._attn_signal = attention_weights(features, target_lagged, period=60)

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

        attn_sig = self._attn_signal[i]
        adx_val = self._adx[i]
        tsi_val = self._tsi_line[i]
        tsi_sig = self._tsi_signal[i]
        if np.isnan(attn_sig) or np.isnan(adx_val) or np.isnan(tsi_val) or np.isnan(tsi_sig):
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

        # 3. Signal exit: TSI direction flip
        if side == 1 and tsi_val < 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and tsi_val > 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry
        if side == 0 and adx_val > self.adx_threshold:
            if attn_sig > 0 and tsi_val > 0 and tsi_val > tsi_sig:
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
            elif attn_sig < 0 and tsi_val < 0 and tsi_val < tsi_sig:
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
        elif side != 0 and self._should_add(price, atr_val, tsi_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if self.direction == 1:
                    context.buy(add_lots)
                else:
                    context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, tsi_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1:
            if price < self.entry_price + atr_val:
                return False
            if tsi_val <= 0:
                return False
        elif self.direction == -1:
            if price > self.entry_price - atr_val:
                return False
            if tsi_val >= 0:
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
