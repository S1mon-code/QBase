"""
Boss Strategy v1 — EMA Breakout Momentum
=========================================
Enter when price breaks ABOVE a rising EMA cluster with volume confirmation.
LONG ONLY. Supports scale-in (0-3).

Usage:
    ./run.sh strategies/boss/v1.py --symbols AG --freq 4h --start 2022
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volatility.atr import atr
from indicators.trend.ema import ema
from indicators.volume.volume_momentum import volume_momentum

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BossV1(TimeSeriesStrategy):
    """
    策略简介：价格向上突破EMA簇（快线在慢线之上）时入场，配合成交量动量确认。
    交易哲学：不等回调，突破即入。当价格从下方穿越上升中的EMA(20)且EMA(20)>EMA(50)，
              表明动能释放，配合成交量动量>阈值确认突破有效性。
    使用指标：
      - EMA(20): 快速趋势线，突破入场参考
      - EMA(50): 慢速趋势方向过滤
      - Volume Momentum(14): 成交量动量，>阈值表示量能配合
      - ATR(14): 止损距离
    进场条件（做多）：
      1. Close > EMA(20)（价格在快线之上）
      2. Close > EMA(50)（价格在慢线之上）
      3. EMA(20) > EMA(50)（均线多头排列）
      4. 最近3根bar内价格曾从下方穿越EMA(20)（刚突破）
      5. Volume Momentum > vol_mom_thresh（量能确认）
    出场条件：
      - Close < EMA(50)（跌破慢线，趋势破坏）
      - ATR追踪止损
      - 分层止盈（3ATR / 5ATR）
    优点：突破入场抓趋势起点；量能过滤减少假突破；条件相对宽松产生更多交易
    缺点：突破后可能回调导致短期浮亏；震荡市假突破仍会发生
    """
    name = "boss_v1"
    warmup = 200
    freq = "4h"

    # Tunable parameters (<=5)
    ema_fast: int = 20
    ema_slow: int = 50
    vol_mom_thresh: float = 1.0
    atr_stop_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._ema_fast = None
        self._ema_slow = None
        self._vol_mom = None
        self._atr = None
        self._avg_vol = None

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

        self._ema_fast = ema(closes, self.ema_fast)
        self._ema_slow = ema(closes, self.ema_slow)
        self._vol_mom = volume_momentum(volumes, period=14)
        self._atr = atr(highs, lows, closes, period=14)

        # Precompute: did close cross above EMA fast from below within last 3 bars?
        n = len(closes)
        self._recent_crossover = np.zeros(n, dtype=bool)
        for j in range(1, n):
            if closes[j] > self._ema_fast[j] and not np.isnan(self._ema_fast[j]):
                # Check if any of the last 3 bars had close <= ema_fast
                for k in range(max(0, j - 3), j):
                    if not np.isnan(self._ema_fast[k]) and closes[k] <= self._ema_fast[k]:
                        self._recent_crossover[j] = True
                        break

        # avg volume for filter
        cumsum = np.cumsum(np.insert(volumes, 0, 0.0))
        self._avg_vol = np.full(n, np.nan)
        if n >= 20:
            self._avg_vol[20:] = (cumsum[21:n + 1] - cumsum[1:n - 19]) / 20

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        # Filters
        if context.is_rollover:
            return
        if not np.isnan(self._avg_vol[i]) and context.volume < self._avg_vol[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        ema_f = self._ema_fast[i]
        ema_s = self._ema_slow[i]
        vol_mom = self._vol_mom[i]
        if np.isnan(ema_f) or np.isnan(ema_s) or np.isnan(vol_mom):
            return

        self.bars_since_last_scale += 1

        # 1. STOP LOSS
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

        # 2. TIERED PROFIT-TAKING
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

        # 3. SIGNAL EXIT: price below slow EMA → trend broken
        if side == 1:
            if price < ema_s:
                context.close_long()
                self._reset_state()
                return

        # 4. ENTRY: EMA breakout momentum
        if side == 0:
            # Price above both EMAs
            if price <= ema_f or price <= ema_s:
                return
            # EMA cluster bullish
            if ema_f <= ema_s:
                return
            # Recent crossover above EMA fast (within last 3 bars)
            if not self._recent_crossover[i]:
                return
            # Volume momentum confirmation
            if vol_mom < self.vol_mom_thresh:
                return

            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. SCALE-IN
        elif side == 1 and self._should_add(i, price, atr_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, i, price, atr_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        # Strategy-specific: EMA cluster still bullish + volume momentum still strong
        ema_f = self._ema_fast[i]
        ema_s = self._ema_slow[i]
        if np.isnan(ema_f) or np.isnan(ema_s) or ema_f <= ema_s:
            return False
        if price <= ema_f:
            return False
        vol_mom = self._vol_mom[i]
        if np.isnan(vol_mom) or vol_mom < self.vol_mom_thresh * 0.8:
            return False
        return True

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
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
