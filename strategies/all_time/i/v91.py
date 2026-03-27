import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.i.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.volatility.ttm_squeeze import ttm_squeeze
from indicators.momentum.rsi import rsi
from indicators.volume.mfi import mfi
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV91(TimeSeriesStrategy):
    """
    策略简介：TTM Squeeze释放 + RSI动量方向 + MFI资金流向的4h突破策略。

    使用指标：
    - TTM Squeeze: 释放时进场，momentum方向决定多空
    - RSI(14): >50做多确认，<50做空确认
    - MFI(14): >50资金流入，<50资金流出
    - ATR(14): 止损距离计算

    进场条件（做多）：Squeeze释放 + momentum>0 + RSI>50 + MFI>50
    进场条件（做空）：Squeeze释放 + momentum<0 + RSI<50 + MFI<50

    出场条件：ATR追踪止损 / 分层止盈 / Squeeze重新激活

    优点：三重确认减少假突破，MFI验证资金参与度
    缺点：条件过多可能错过快速突破
    """
    name = "i_alltime_v91"
    warmup = 400
    freq = "4h"

    squeeze_period: int = 20
    rsi_period: int = 14
    rsi_threshold: float = 50.0
    mfi_threshold: float = 50.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._squeeze_on = None
        self._momentum = None
        self._rsi = None
        self._mfi = None
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

        self._squeeze_on, self._momentum = ttm_squeeze(highs, lows, closes, period=self.squeeze_period)
        self._rsi = rsi(closes, self.rsi_period)
        self._mfi = mfi(highs, lows, closes, volumes, period=14)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        sq_on = self._squeeze_on[i]
        mom = self._momentum[i]
        rsi_val = self._rsi[i]
        mfi_val = self._mfi[i]
        if np.isnan(atr_val) or np.isnan(mom) or np.isnan(rsi_val) or np.isnan(mfi_val):
            return

        self.bars_since_last_scale += 1

        prev_sq = self._squeeze_on[i - 1] if i > 0 else True
        squeeze_release = prev_sq and not sq_on

        # ── 1. 止损 ──
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

        # ── 2. 分层止盈 ──
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
        elif side == -1 and self.entry_price > 0:
            profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_3atr = True
                return

        # ── 3. 信号退出 ──
        if side != 0 and sq_on:
            if side == 1:
                context.close_long()
            else:
                context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场 ──
        if side == 0 and squeeze_release:
            if mom > 0 and rsi_val > self.rsi_threshold and mfi_val > self.mfi_threshold:
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
            elif mom < 0 and rsi_val < self.rsi_threshold and mfi_val < self.mfi_threshold:
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

        # ── 5. 加仓 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and mom > 0 and rsi_val > self.rsi_threshold):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and mom < 0 and rsi_val < self.rsi_threshold):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.sell(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.direction = 0
