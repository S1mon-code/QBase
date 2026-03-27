"""
Boss Strategy v8 — EMA Ribbon Fan-out
======================================
When multiple EMAs align and fan out, it signals a powerful trend launch.
LONG ONLY. Supports scale-in (0-3).

Usage:
    ./run.sh strategies/boss/v8.py --symbols AG --freq 1h --start 2022
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
from indicators.volume.klinger import klinger

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BossV8(TimeSeriesStrategy):
    """
    策略简介：EMA带状线（多条EMA）扇出时入场，捕捉趋势加速的起始点。Klinger确认聪明资金。
    交易哲学：单根均线可以骗你，但6根均线同时排列、间距扩大——这是趋势真正启动的信号。
              带宽从收敛到扇出的过程，反映了市场从犹豫到共识的转变。
    使用指标：
      - EMA Ribbon (8,13,21,34,55,89): 6条EMA构成的带状线
      - Ribbon Width: 最快EMA与最慢EMA的差值（手动计算）
      - Klinger(34,55,13): 量价振荡器，线上穿信号线=买入压力
      - ATR(14): 止损距离
    进场条件（做多）：
      1. 所有EMA正确排列（EMA8 > EMA13 > EMA21 > EMA34 > EMA55 > EMA89）
      2. Ribbon Width 扩张中（当前宽度 > 10根bar前宽度 * width_expand）
      3. Klinger线 > 信号线（聪明资金买入）
    出场条件：
      - Ribbon Width 压缩（当前宽度 < 5根bar前宽度 * width_compress）
      - ATR追踪止损
      - 分层止盈（3ATR / 5ATR）
    优点：多均线共振是最可靠的趋势确认之一；Klinger增加资金面维度
    缺点：入场较晚（需要全部排列+扇出）；1h频率下信号较多需仔细管理
    """
    name = "boss_v8"
    warmup = 500
    freq = "1h"

    # Tunable parameters (<=5)
    ribbon_base: int = 8
    width_expand: float = 1.5
    width_compress: float = 0.7
    atr_stop_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._emas = None  # list of 6 EMA arrays
        self._ribbon_width = None
        self._klinger_line = None
        self._klinger_signal = None
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

        # Build EMA ribbon from Fibonacci-like periods scaled to ribbon_base
        # Periods: base, base*~1.6, base*~2.6, base*~4.25, base*~6.9, base*~11.1
        periods = [
            self.ribbon_base,
            int(self.ribbon_base * 1.625),   # ~13
            int(self.ribbon_base * 2.625),   # ~21
            int(self.ribbon_base * 4.25),    # ~34
            int(self.ribbon_base * 6.875),   # ~55
            int(self.ribbon_base * 11.125),  # ~89
        ]
        self._emas = [ema(closes, p) for p in periods]
        # Ribbon width = fastest EMA - slowest EMA
        self._ribbon_width = self._emas[0] - self._emas[-1]

        self._klinger_line, self._klinger_signal = klinger(highs, lows, closes, volumes)
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

        # Read ribbon values
        ema_vals = [e[i] for e in self._emas]
        if any(np.isnan(v) for v in ema_vals):
            return

        rw = self._ribbon_width[i]
        kl = self._klinger_line[i]
        ks = self._klinger_signal[i]
        if np.isnan(rw) or np.isnan(kl) or np.isnan(ks):
            return

        self.bars_since_last_scale += 1

        # Check ribbon alignment: EMA8 > EMA13 > ... > EMA89
        ribbon_aligned = all(ema_vals[j] > ema_vals[j + 1] for j in range(len(ema_vals) - 1))

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

        # 3. SIGNAL EXIT: ribbon width compresses
        if side == 1:
            if i >= 5:
                rw_prev = self._ribbon_width[i - 5]
                if not np.isnan(rw_prev) and rw_prev > 0 and rw < rw_prev * self.width_compress:
                    context.close_long()
                    self._reset_state()
                    return

        # 4. ENTRY: ribbon aligned + width expanding + Klinger bullish
        if side == 0:
            if not ribbon_aligned:
                return
            if rw <= 0:
                return
            # Width expanding
            if i < 10:
                return
            rw_prev = self._ribbon_width[i - 10]
            if np.isnan(rw_prev) or rw_prev <= 0:
                return
            if rw < rw_prev * self.width_expand:
                return
            # Klinger line > signal
            if kl <= ks:
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
        # Strategy-specific: ribbon still aligned + Klinger still bullish
        ema_vals = [e[i] for e in self._emas]
        if any(np.isnan(v) for v in ema_vals):
            return False
        if not all(ema_vals[j] > ema_vals[j + 1] for j in range(len(ema_vals) - 1)):
            return False
        if self._klinger_line[i] <= self._klinger_signal[i]:
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
