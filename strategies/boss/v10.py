"""
Boss Strategy v10 — Vol Regime + SMA Cross
============================================
Wait for volatility compression→expansion shift, then follow the SMA signal.
LONG ONLY. Supports scale-in (0-3).

Usage:
    ./run.sh strategies/boss/v10.py --symbols AG --freq daily --start 2022
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
from indicators.volatility.historical_vol import historical_volatility
from indicators.trend.sma import sma
from indicators.volume.volume_spike import volume_spike

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BossV10(TimeSeriesStrategy):
    """
    策略简介：波动率政权转换（压缩→扩张）配合SMA金叉，捕捉趋势爆发的起始点。
    交易哲学：大趋势几乎总是从波动率压缩后爆发。低波动率=市场蓄力，
              波动率突然扩张+均线多头排列=趋势正式启动。成交量放大确认这不是假突破。
    使用指标：
      - Historical Volatility(20): 历史波动率，检测压缩→扩张
      - SMA(20) / SMA(60): 均线交叉，确认趋势方向
      - Volume Spike: 成交量放大确认政权转变
      - ATR(14): 止损距离
    进场条件（做多）：
      1. 波动率扩张（hvol[i] > hvol[i-10] * vol_expand）
      2. SMA(20) > SMA(60)（均线多头排列）
      3. SMA(20) 上升中（SMA20[i] > SMA20[i-3]）
      4. 成交量放大确认（最近3根bar有volume spike）
    出场条件：
      - SMA(20) < SMA(60)（均线死叉 → 趋势结束）
      - ATR追踪止损
      - 分层止盈（3ATR / 5ATR）
    优点：波动率政权过滤是最强的趋势起点捕捉工具；多重确认降低假信号
    缺点：波动率扩张不一定是向上的（需SMA过滤方向）；信号较稀疏
    """
    name = "boss_v10"
    warmup = 200
    freq = "daily"

    # Tunable parameters (<=5)
    hvol_period: int = 20
    sma_fast: int = 20
    sma_slow: int = 60
    vol_expand: float = 1.3
    atr_stop_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._hvol = None
        self._sma_fast = None
        self._sma_slow = None
        self._vol_spikes = None
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

        self._hvol = historical_volatility(closes, period=self.hvol_period)
        self._sma_fast = sma(closes, self.sma_fast)
        self._sma_slow = sma(closes, self.sma_slow)
        self._vol_spikes = volume_spike(volumes, period=20, threshold=2.0)
        self._atr = atr(highs, lows, closes, period=14)

        n = len(volumes)
        cumsum = np.cumsum(np.insert(volumes, 0, 0.0))
        self._avg_vol = np.full(n, np.nan)
        if n >= 20:
            self._avg_vol[20:] = (cumsum[21:n + 1] - cumsum[1:n - 19]) / 20

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_vol[i]) and context.volume < self._avg_vol[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        hvol_now = self._hvol[i]
        sma_f = self._sma_fast[i]
        sma_s = self._sma_slow[i]
        if np.isnan(hvol_now) or np.isnan(sma_f) or np.isnan(sma_s):
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

        # 3. SIGNAL EXIT: SMA death cross → trend over
        if side == 1:
            if sma_f < sma_s:
                context.close_long()
                self._reset_state()
                return

        # 4. ENTRY: vol expanding + SMA golden cross + SMA rising + volume spike
        if side == 0:
            if i < 10:
                return
            # Volatility expansion
            hvol_prev = self._hvol[i - 10]
            if np.isnan(hvol_prev) or hvol_prev <= 0:
                return
            if hvol_now < hvol_prev * self.vol_expand:
                return
            # SMA golden cross
            if sma_f <= sma_s:
                return
            # SMA(20) rising
            if i < 3:
                return
            sma_f_prev = self._sma_fast[i - 3]
            if np.isnan(sma_f_prev) or sma_f <= sma_f_prev:
                return
            # Volume spike in last 3 bars
            start_idx = max(0, i - 2)
            if not np.any(self._vol_spikes[start_idx:i + 1]):
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
        # Strategy-specific: SMA golden cross still valid + vol still elevated
        if self._sma_fast[i] <= self._sma_slow[i]:
            return False
        if i >= 10:
            hvol_prev = self._hvol[i - 10]
            hvol_now = self._hvol[i]
            if not np.isnan(hvol_prev) and not np.isnan(hvol_now):
                if hvol_now < hvol_prev:
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
