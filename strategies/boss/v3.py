"""
Boss Strategy v3 — Bollinger Band Walk
=======================================
In strong trends, price "walks" along the upper Bollinger Band.
LONG ONLY. Supports scale-in (0-3).

Usage:
    ./run.sh strategies/boss/v3.py --symbols AG --freq daily --start 2022
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
from indicators.volatility.bollinger import bollinger_bands
from indicators.volume.cmf import cmf

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BossV3(TimeSeriesStrategy):
    """
    策略简介：捕捉价格沿布林带上轨"行走"的强趋势行情，布林带宽扩张确认趋势加速。
    交易哲学：当价格连续收在布林带上方，这不是超买，而是强势信号。带宽扩张说明趋势在加速。
    使用指标：
      - Bollinger Bands(20, 2.0): 上轨突破 + 中轨止盈参考
      - BB Width: 带宽扩张确认趋势加速（手动计算 upper-lower）
      - CMF(20): 资金流量指标，> 0 表示买方主导
      - ATR(14): 止损距离
    进场条件（做多）：
      1. 收盘价 > 布林上轨
      2. 前一根bar收盘也 > 上轨（连续两根=确认走势）
      3. BB Width 扩张中（当前宽度 > 5根bar前宽度）
      4. CMF > 0（资金流入）
    出场条件：
      - 收盘价 < 布林中轨 → 回归均值，趋势暂停
      - ATR追踪止损
      - 分层止盈（3ATR / 5ATR）
    优点：只在最强势的趋势中交易；带宽过滤减少假突破
    缺点：入场偏晚（需两根bar确认）；急速回调可能来不及出场
    """
    name = "boss_v3"
    warmup = 120
    freq = "daily"

    # Tunable parameters (<=5)
    bb_period: int = 20
    bb_std: float = 2.0
    width_lookback: int = 5
    atr_stop_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._bb_upper = None
        self._bb_middle = None
        self._bb_lower = None
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

        upper, middle, lower = bollinger_bands(closes, self.bb_period, self.bb_std)
        self._bb_upper = upper
        self._bb_middle = middle
        self._bb_lower = lower
        self._bb_width = upper - lower  # raw width
        self._cmf = cmf(highs, lows, closes, volumes, period=20)
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

        bb_upper = self._bb_upper[i]
        bb_mid = self._bb_middle[i]
        cmf_val = self._cmf[i]
        width_now = self._bb_width[i]
        if np.isnan(bb_upper) or np.isnan(bb_mid) or np.isnan(cmf_val) or np.isnan(width_now):
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

        # 3. SIGNAL EXIT: close below middle BB → trend pause
        if side == 1:
            if price < bb_mid:
                context.close_long()
                self._reset_state()
                return

        # 4. ENTRY: BB walk — consecutive closes above upper BB + width expanding + CMF > 0
        if side == 0:
            if i < self.width_lookback + 1:
                return
            # Current close > upper BB
            if price <= bb_upper:
                return
            # Previous close also > upper BB
            prev_close = context.get_full_close_array()[i - 1]
            prev_upper = self._bb_upper[i - 1]
            if np.isnan(prev_upper) or prev_close <= prev_upper:
                return
            # BB width expanding
            width_prev = self._bb_width[i - self.width_lookback]
            if np.isnan(width_prev) or width_now <= width_prev:
                return
            # CMF positive
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
        # Strategy-specific: still above upper BB and width still expanding
        if price <= self._bb_upper[i]:
            return False
        if i >= self.width_lookback:
            w_prev = self._bb_width[i - self.width_lookback]
            if not np.isnan(w_prev) and self._bb_width[i] <= w_prev:
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
