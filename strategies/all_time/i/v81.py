import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.i.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.trend.donchian import donchian
from indicators.trend.adx import adx
from indicators.volume.volume_spike import volume_spike
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV81(TimeSeriesStrategy):
    """
    策略简介：Donchian(20) 通道突破 + ADX 趋势强度上升 + 成交量飙升确认的突破策略。

    使用指标：
    - Donchian(20): 价格突破上下轨产生入场信号
    - ADX(14): 趋势强度确认，ADX上升表示趋势正在形成
    - Volume Spike: 成交量相对均量的倍数，确认突破有效性
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - 价格突破Donchian上轨
    - ADX > adx_threshold 且 ADX 上升（当前 > 前一根）
    - 成交量 > vol_spike_mult 倍均量

    进场条件（做空）：
    - 价格跌破Donchian下轨
    - ADX > adx_threshold 且 ADX 上升
    - 成交量 > vol_spike_mult 倍均量

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR / 5ATR）
    - 价格回到通道内部（信号反转）

    优点：经典突破系统，量价确认减少假突破
    缺点：震荡市频繁假突破，ADX滞后于价格
    """
    name = "i_alltime_v81"
    warmup = 250
    freq = "daily"

    don_period: int = 20
    adx_period: int = 14
    adx_threshold: float = 20.0
    vol_spike_mult: float = 1.5
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._don_upper = None
        self._don_lower = None
        self._adx_val = None
        self._vol_spike = None
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

        self._don_upper, self._don_lower, _ = donchian(highs, lows, self.don_period)
        adx_result = adx(highs, lows, closes, period=self.adx_period)
        self._adx_val = adx_result[0]
        self._vol_spike = volume_spike(volumes, period=20)
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
        don_up = self._don_upper[i]
        don_low = self._don_lower[i]
        adx_now = self._adx_val[i]
        vs = self._vol_spike[i]
        if np.isnan(atr_val) or np.isnan(don_up) or np.isnan(don_low) or np.isnan(adx_now) or np.isnan(vs):
            return

        adx_prev = self._adx_val[i - 1] if i > 0 else np.nan
        if np.isnan(adx_prev):
            return

        self.bars_since_last_scale += 1

        adx_rising = adx_now > adx_prev and adx_now > self.adx_threshold
        vol_confirmed = vs > self.vol_spike_mult

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

        # ── 3. 信号退出：价格回到通道内 ──
        if side == 1 and price < don_low:
            context.close_long()
            self._reset_state()
        elif side == -1 and price > don_up:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场 ──
        if side == 0:
            if price > don_up and adx_rising and vol_confirmed:
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
            elif price < don_low and adx_rising and vol_confirmed:
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
                    and adx_rising):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and adx_rising):
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
