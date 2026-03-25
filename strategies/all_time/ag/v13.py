import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.trend.ema import ema

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _amihud_illiquidity(closes, volumes, period=20):
    """Amihud illiquidity ratio: |return| / dollar_volume (rolling average).

    High illiquidity = thin market, moves are unreliable.
    Low illiquidity = liquid market, trend moves are reliable.
    Returns (illiq, illiq_zscore) arrays.
    """
    n = len(closes)
    illiq = np.full(n, np.nan, dtype=np.float64)
    illiq_z = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return illiq, illiq_z

    daily_illiq = np.full(n, np.nan)
    for i in range(1, n):
        ret = abs(closes[i] / closes[i - 1] - 1.0)
        dvol = closes[i] * max(volumes[i], 1.0)
        daily_illiq[i] = ret / dvol * 1e9  # Scale for readability

    for i in range(period + 1, n):
        window = daily_illiq[i - period + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        illiq[i] = np.mean(valid)

    # Z-score of illiquidity
    for i in range(period * 3, n):
        long_window = illiq[i - period * 3:i + 1]
        valid = long_window[~np.isnan(long_window)]
        if len(valid) < 10:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid)
        if sigma < 1e-15:
            illiq_z[i] = 0.0
        else:
            illiq_z[i] = (illiq[i] - mu) / sigma

    return illiq, illiq_z


class StrategyV13(TimeSeriesStrategy):
    """
    策略简介：基于Amihud非流动性指标，流动市趋势跟踪，非流动市反转。

    使用指标：
    - Amihud Illiquidity(20): 市场流动性度量
    - EMA(30): 趋势方向
    - ATR(14): 止损距离计算

    进场条件（做多-流动）：illiq_z<0(流动性好), 价格>EMA30
    进场条件（做空-流动）：illiq_z<0(流动性好), 价格<EMA30
    进场条件（做多-非流动）：illiq_z>1.5(非流动), 价格跌>2ATR后反弹
    进场条件（做空-非流动）：illiq_z>1.5(非流动), 价格涨>2ATR后回落

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 流动性状态切换

    优点：区分流动性环境选择不同策略模式
    缺点：非流动性环境下成交可能困难，滑点大
    """
    name = "ag_alltime_v13"
    warmup = 300
    freq = "4h"

    illiq_period: int = 20        # Optuna: 10-40
    liquid_thresh: float = 0.0   # Optuna: -1.0-0.5
    illiq_thresh: float = 1.5    # Optuna: 1.0-2.5
    ema_period: int = 30          # Optuna: 15-50
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._illiq = None
        self._illiq_z = None
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
        self.mode = 0  # 1=trend(liquid), 2=fade(illiquid)

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._atr = atr(highs, lows, closes, period=14)
        self._ema = ema(closes, period=self.ema_period)
        self._illiq, self._illiq_z = _amihud_illiquidity(closes, volumes, period=self.illiq_period)

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
        iz = self._illiq_z[i]
        ema_val = self._ema[i]
        if np.isnan(iz) or np.isnan(ema_val):
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

        # 3. Signal-based exit
        if side != 0:
            if self.mode == 1 and iz > self.illiq_thresh:
                # Switched to illiquid - exit trend
                if side == 1:
                    context.close_long()
                else:
                    context.close_short()
                self._reset_state()
                return
            if self.mode == 2:
                # Fade: take profit near EMA
                if side == 1 and price >= ema_val:
                    context.close_long()
                    self._reset_state()
                    return
                if side == -1 and price <= ema_val:
                    context.close_short()
                    self._reset_state()
                    return

        # 4. Entry
        if side == 0:
            if iz < self.liquid_thresh:
                # Liquid: trend follow
                if price > ema_val:
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
                        self.mode = 1
                elif price < ema_val:
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
                        self.mode = 1
            elif iz > self.illiq_thresh:
                # Illiquid: fade extreme moves
                if price < ema_val - 2.0 * atr_val:
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
                        self.mode = 2
                elif price > ema_val + 2.0 * atr_val:
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
                        self.mode = 2

        # 5. Scale-in (trend mode only)
        elif side != 0 and self.mode == 1 and self._should_add(price, atr_val):
            if self.direction == 1 and iz < self.liquid_thresh:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and iz < self.liquid_thresh:
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
        self.mode = 0
