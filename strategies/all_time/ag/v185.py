import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.regime.trend_persistence import trend_persistence
from indicators.trend.hma import hma
from indicators.microstructure.amihud import amihud_illiquidity

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV185(TimeSeriesStrategy):
    """
    策略简介：Trend Persistence 趋势持续性 + HMA 方向 + Amihud 流动性过滤的多空策略。

    使用指标：
    - Trend Persistence(20, 60): Hurst-like 持续性度量，高值=趋势，低值=均值回归
    - HMA(20): Hull均线方向作为趋势确认，比EMA更平滑
    - Amihud Illiquidity(20): 流动性度量，在正常流动性下交易

    进场条件（做多）：persistence > 阈值 + HMA 上行 + Amihud正常（非极端低流动性）
    进场条件（做空）：persistence > 阈值 + HMA 下行 + Amihud正常

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - HMA方向反转

    优点：Persistence过滤震荡市，HMA滞后小，Amihud避免流动性陷阱
    缺点：Persistence计算需要较长lookback，短期趋势可能错过
    """
    name = "ag_alltime_v185"
    warmup = 500
    freq = "4h"

    persist_max_lag: int = 20
    persist_period: int = 60
    persist_thresh: float = 0.5
    hma_period: int = 20
    amihud_period: int = 20
    amihud_max_pct: float = 95.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._persist = None
        self._hma = None
        self._amihud = None
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
        self._persist = trend_persistence(closes, max_lag=self.persist_max_lag, period=self.persist_period)
        self._hma = hma(closes, period=self.hma_period)
        self._amihud = amihud_illiquidity(closes, volumes, period=self.amihud_period)

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
        persist_val = self._persist[i]
        if np.isnan(persist_val):
            return

        # HMA direction: compare current vs previous
        if i < 1 or np.isnan(self._hma[i]) or np.isnan(self._hma[i - 1]):
            return
        hma_dir = 1 if self._hma[i] > self._hma[i - 1] else -1

        # Amihud check: skip if illiquidity is extreme (above 95th percentile rolling)
        amihud_val = self._amihud[i]
        if np.isnan(amihud_val):
            return
        # Simple rolling percentile for Amihud
        lookback = min(i + 1, 252)
        if lookback > 20:
            recent_amihud = self._amihud[i - lookback + 1:i + 1]
            valid = recent_amihud[~np.isnan(recent_amihud)]
            if len(valid) > 10:
                pct = np.percentile(valid, self.amihud_max_pct)
                if amihud_val > pct:
                    return  # Skip extreme illiquidity

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

        # 3. Signal-based exit: HMA direction reversal
        if side == 1 and hma_dir < 0:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and hma_dir > 0:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and persist_val > self.persist_thresh:
            if hma_dir > 0:
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
            elif hma_dir < 0:
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
            if self.direction == 1 and hma_dir > 0 and persist_val > self.persist_thresh:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and hma_dir < 0 and persist_val > self.persist_thresh:
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
