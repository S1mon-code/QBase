"""
Boss Strategy v10 — Triple MA + Bollinger Width
=================================================
Enter when all 3 MAs align bullish + Bollinger Width expanding + CMF positive.
LONG ONLY. Supports scale-in (0-3).

Usage:
    ./run.sh strategies/boss/v10.py --symbols AG --freq 4h --start 2022
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
from indicators.trend.sma import sma
from indicators.volatility.bollinger import bollinger_width
from indicators.volume.cmf import cmf

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BossV10(TimeSeriesStrategy):
    """
    策略简介：三重均线多头排列+布林带宽度扩张+CMF资金流确认的趋势跟踪策略。
    交易哲学：当SMA(20)>SMA(50)>SMA(100)完美排列时趋势明确，
              布林带宽度扩张表明波动率增加支持趋势延续，
              CMF>0确认买方资金持续流入。三者共振=高概率趋势信号。
              4h频率替代daily产生更多交易机会。
    使用指标：
      - SMA(20): 快速均线
      - SMA(50): 中速均线
      - SMA(100): 慢速均线（三线排列确认趋势强度）
      - Bollinger Width(20): 布林带宽度，扩张表示波动率释放
      - CMF(20): Chaikin资金流量，>0表示买方主导
      - ATR(14): 止损距离
    进场条件（做多）：
      1. SMA(20) > SMA(50) > SMA(100)（三重均线多头排列）
      2. BB Width[i] > BB Width[i-10]（布林带宽度在扩张）
      3. Close > SMA(20)（价格在快线之上）
      4. CMF > 0（资金流入确认）
    出场条件：
      - SMA(20) < SMA(50)（快线跌破中线，趋势弱化）
      - ATR追踪止损
      - 分层止盈（3ATR / 5ATR）
    优点：三重均线排列过滤大部分噪音；BB Width捕捉波动率扩张；CMF确认资金面
    缺点：三线排列滞后，可能错过趋势前半段；震荡市均线缠绕频繁触发假信号
    """
    name = "boss_v10"
    warmup = 300
    freq = "4h"

    # Tunable parameters (<=5)
    sma_fast: int = 20
    sma_mid: int = 50
    sma_slow: int = 100
    bb_period: int = 20
    atr_stop_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._sma_fast = None
        self._sma_mid = None
        self._sma_slow = None
        self._bb_width = None
        self._cmf = None
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

        self._sma_fast = sma(closes, self.sma_fast)
        self._sma_mid = sma(closes, self.sma_mid)
        self._sma_slow = sma(closes, self.sma_slow)
        self._bb_width = bollinger_width(closes, period=self.bb_period)
        self._cmf = cmf(highs, lows, closes, volumes, period=20)
        self._atr = atr(highs, lows, closes, period=14)

        # avg volume
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

        sma_f = self._sma_fast[i]
        sma_m = self._sma_mid[i]
        sma_s = self._sma_slow[i]
        bb_w = self._bb_width[i]
        cmf_val = self._cmf[i]
        if np.isnan(sma_f) or np.isnan(sma_m) or np.isnan(sma_s):
            return
        if np.isnan(bb_w) or np.isnan(cmf_val):
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

        # 3. SIGNAL EXIT: SMA(20) < SMA(50) → trend weakening
        if side == 1:
            if sma_f < sma_m:
                context.close_long()
                self._reset_state()
                return

        # 4. ENTRY: Triple MA alignment + BB Width expanding + CMF positive
        if side == 0:
            # Triple MA bullish alignment
            if not (sma_f > sma_m > sma_s):
                return
            # Price above fast SMA
            if price <= sma_f:
                return
            # Bollinger Width expanding (current > 10 bars ago)
            if i < 10:
                return
            bb_w_prev = self._bb_width[i - 10]
            if np.isnan(bb_w_prev) or bb_w <= bb_w_prev:
                return
            # CMF positive (buying pressure)
            if cmf_val <= 0:
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
        # Strategy-specific: triple MA still aligned + CMF still positive
        sma_f = self._sma_fast[i]
        sma_m = self._sma_mid[i]
        sma_s = self._sma_slow[i]
        if np.isnan(sma_f) or np.isnan(sma_m) or np.isnan(sma_s):
            return False
        if not (sma_f > sma_m > sma_s):
            return False
        cmf_val = self._cmf[i]
        if np.isnan(cmf_val) or cmf_val <= 0:
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
