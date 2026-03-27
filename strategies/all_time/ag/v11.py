import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.volume.obv import obv
from indicators.trend.ema import ema

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _smart_money_index(closes, volumes, period=20):
    """Smart Money Index: compares first-hour and last-hour behavior.

    Approximation: uses volume-weighted price action. Smart money acts
    in the last portion of the day, dumb money in the first portion.
    We split each period window and compare early vs late behavior.

    Returns (smi, divergence_zscore) arrays.
    divergence_zscore > 2 = smart money diverging bullish from price
    divergence_zscore < -2 = smart money diverging bearish from price
    """
    n = len(closes)
    smi = np.full(n, np.nan, dtype=np.float64)
    div_z = np.full(n, np.nan, dtype=np.float64)

    if n < period * 2:
        return smi, div_z

    # Smart money proxy: volume-weighted close position (high vol = smart money)
    # Compare OBV trend vs price trend
    obv_arr = obv(closes, volumes)

    # Normalize OBV and price to same scale over rolling window
    for i in range(period * 2, n):
        price_window = closes[i - period:i + 1]
        obv_window = obv_arr[i - period:i + 1]

        p_min, p_max = np.min(price_window), np.max(price_window)
        o_min, o_max = np.min(obv_window), np.max(obv_window)

        if p_max - p_min < 1e-9 or o_max - o_min < 1e-9:
            smi[i] = 0.0
            div_z[i] = 0.0
            continue

        # Normalize to [0,1]
        p_norm = (closes[i] - p_min) / (p_max - p_min)
        o_norm = (obv_arr[i] - o_min) / (o_max - o_min)

        # SMI = OBV normalized - Price normalized
        smi[i] = o_norm - p_norm

        # Rolling z-score of SMI divergence
        smi_window = smi[i - period:i + 1]
        valid = smi_window[~np.isnan(smi_window)]
        if len(valid) < 5:
            div_z[i] = 0.0
            continue
        mu = np.mean(valid)
        sigma = np.std(valid)
        if sigma < 1e-12:
            div_z[i] = 0.0
        else:
            div_z[i] = (smi[i] - mu) / sigma

    return smi, div_z


class StrategyV11(TimeSeriesStrategy):
    """
    策略简介：基于Smart Money Index背离信号跟随聪明钱的交易策略。

    使用指标：
    - Smart Money Index(20): OBV与价格的标准化背离
    - EMA(30): 趋势方向过滤
    - ATR(14): 止损距离计算

    进场条件（做多）：SMI背离z-score>2.0（OBV比价格更强）
    进场条件（做空）：SMI背离z-score<-2.0（OBV比价格更弱）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 背离信号消失（z-score回到[-1,1]）

    优点：利用量价背离捕捉聪明钱方向，领先于纯价格信号
    缺点：OBV在期货市场中可能受换月影响，背离持续时间不确定
    """
    name = "ag_alltime_v11"
    warmup = 250
    freq = "4h"

    smi_period: int = 20          # Optuna: 10-40
    div_thresh: float = 2.0      # Optuna: 1.5-3.0
    ema_period: int = 30          # Optuna: 15-50
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._smi = None
        self._div_z = None
        self._ema = None
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
        self._ema = ema(closes, period=self.ema_period)
        self._smi, self._div_z = _smart_money_index(closes, volumes, period=self.smi_period)

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
        div_z = self._div_z[i]
        if np.isnan(div_z):
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

        # 3. Signal-based exit: divergence faded
        if side == 1 and div_z < -1.0:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and div_z > 1.0:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if div_z > self.div_thresh:
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
            elif div_z < -self.div_thresh:
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
            if self.direction == 1 and div_z > 1.0:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and div_z < -1.0:
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
