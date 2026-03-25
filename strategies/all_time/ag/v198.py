import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.boosting_signal import gradient_boost_signal
from indicators.trend.psar import psar
from indicators.volume.volume_spike import volume_spike
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV198(TimeSeriesStrategy):
    """
    策略简介：Gradient Boosting方向预测 + PSAR翻转 + Volume Spike确认的ML趋势策略。

    使用指标：
    - Gradient Boost(120, 20): 基于RSI/ADX/ATR特征预测方向概率
    - PSAR(0.02, 0.02, 0.2): 趋势翻转确认
    - Volume Spike(20, 2.0): 放量确认突破有效性
    - ATR(14): 止损距离计算

    进场条件（做多）：GBT预测概率>0.6 + PSAR翻多 + 成交量放大
    进场条件（做空）：GBT预测概率<0.4 + PSAR翻空 + 成交量放大

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - PSAR反向翻转

    优点：ML概率过滤提高PSAR信号质量，Volume Spike确认资金参与
    缺点：GBT可能过拟合历史模式，重训练增加计算开销
    """
    name = "ag_alltime_v198"
    warmup = 300
    freq = "4h"

    gbt_period: int = 120         # Optuna: 80-160
    gbt_estimators: int = 20      # Optuna: 10-30
    gbt_prob_thresh: float = 0.6  # Optuna: 0.55-0.70
    psar_af_start: float = 0.02   # Optuna: 0.01-0.04
    vol_spike_mult: float = 2.0   # Optuna: 1.5-3.0
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._gbt_prob = None
        self._gbt_conf = None
        self._psar_val = None
        self._psar_dir = None
        self._vol_spike = None
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

        # Build features for GBT
        rsi_arr = rsi(closes, 14)
        adx_arr = adx(highs, lows, closes, 14)
        atr_arr = atr(highs, lows, closes, 14)
        self._atr = atr_arr

        features = np.column_stack([rsi_arr, adx_arr, atr_arr])
        self._gbt_prob, self._gbt_conf = gradient_boost_signal(
            closes, features, period=self.gbt_period, n_estimators=self.gbt_estimators)

        self._psar_val, self._psar_dir = psar(
            highs, lows, self.psar_af_start, 0.02, 0.2)
        self._vol_spike = volume_spike(volumes, period=20, threshold=self.vol_spike_mult)

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
        prob = self._gbt_prob[i]
        pdir = self._psar_dir[i]
        v_spike = self._vol_spike[i]

        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(prob) or np.isnan(pdir):
            return

        prev_dir = self._psar_dir[i - 1] if i > 0 else np.nan
        if np.isnan(prev_dir):
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

        # 3. Signal exit: PSAR flip
        if side == 1 and pdir == -1 and prev_dir == 1:
            context.close_long()
            self._reset_state()
        elif side == -1 and pdir == 1 and prev_dir == -1:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: GBT probability + PSAR flip + volume spike
        if side == 0:
            if (prob > self.gbt_prob_thresh and prev_dir == -1 and pdir == 1
                    and v_spike):
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
            elif (prob < (1 - self.gbt_prob_thresh) and prev_dir == 1 and pdir == -1
                    and v_spike):
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
            if self.direction == 1 and prob > self.gbt_prob_thresh and pdir == 1:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and prob < (1 - self.gbt_prob_thresh) and pdir == -1:
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
