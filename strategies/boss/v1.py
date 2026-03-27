"""
Boss Strategy v1 — Pullback to Rising EMA
==========================================
Wait for a confirmed uptrend, then buy the dip back to EMA(20).
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
from indicators.volume.obv import obv

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BossV1(TimeSeriesStrategy):
    """
    策略简介：在确认的上升趋势中，等待价格回调至快速均线附近买入，顺势而为。
    交易哲学：不追突破，等回调。趋势已确立后，回调到均线支撑是最佳入场时机。
    使用指标：
      - EMA(20): 快速趋势线 + 回调入场参考
      - EMA(60): 慢速趋势方向过滤
      - OBV: 资金流向确认（OBV新高=资金持续流入）
      - ATR(14): 止损距离 + 回调幅度衡量
    进场条件（做多）：
      1. EMA(20) > EMA(60)（趋势向上）
      2. EMA(20) 当前值 > 3根bar前值（均线上升中）
      3. 价格在 EMA(20) 附近（距离 < pullback_atr_mult * ATR）
      4. 价格在 EMA(20) 之上（不买破位的）
      5. OBV 创20周期新高（资金确认）
    出场条件：
      - ATR追踪止损
      - 分层止盈（3ATR / 5ATR）
      - 价格收盘低于 EMA(60) → 趋势破坏，全部平仓
    优点：顺势回调入场，胜率较高；OBV过滤虚假回调
    缺点：强趋势中回调幅度不够可能错过行情；震荡市频繁触发假信号
    """
    name = "boss_v1"
    warmup = 200
    freq = "4h"

    # Tunable parameters (<=5)
    ema_fast: int = 20
    ema_slow: int = 60
    pullback_atr_mult: float = 0.5
    atr_stop_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._ema_fast = None
        self._ema_slow = None
        self._obv = None
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
        self._obv = obv(closes, volumes)
        self._atr = atr(highs, lows, closes, period=14)

        # avg volume for filter
        n = len(volumes)
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
        obv_val = self._obv[i]
        if np.isnan(ema_f) or np.isnan(ema_s) or np.isnan(obv_val):
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

        # 4. ENTRY: pullback to rising EMA in uptrend
        if side == 0:
            # Trend conditions
            if ema_f <= ema_s:
                return
            # EMA(20) must be rising
            if i < 3:
                return
            ema_f_prev = self._ema_fast[i - 3]
            if np.isnan(ema_f_prev) or ema_f <= ema_f_prev:
                return
            # Price near EMA(20) from above
            dist = price - ema_f
            if dist < 0 or dist > self.pullback_atr_mult * atr_val:
                return
            # OBV making new 20-period high
            if i < 20:
                return
            obv_window = self._obv[max(0, i - 20):i]
            if obv_val <= np.nanmax(obv_window):
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
        # Strategy-specific: EMA fast still above slow and price near EMA fast
        ema_f = self._ema_fast[i]
        ema_s = self._ema_slow[i]
        if ema_f <= ema_s:
            return False
        dist = price - ema_f
        if dist < 0 or dist > self.pullback_atr_mult * atr_val * 1.5:
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
