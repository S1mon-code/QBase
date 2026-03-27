import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.regime.vol_regime_markov import vol_regime_simple
from indicators.regime.regime_switch_speed import switch_speed
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class RegimeDurationSupertrend(TimeSeriesStrategy):
    """
    策略简介：状态持续时间 + Supertrend。仅在新鲜状态（刚切换）中跟随 Supertrend。

    使用指标：
    - vol_regime_simple(60): 波动率状态
    - switch_speed: 当前状态持续时间
    - supertrend(10, 3.0): 趋势方向和止损线
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - 当前状态持续 < 20 bar（新鲜状态）
    - Supertrend direction = 1（看涨）

    进场条件（做空）：
    - 当前状态持续 < 20 bar
    - Supertrend direction = -1（看跌）

    出场条件：
    - ATR 追踪止损 / 分层止盈
    - Supertrend 方向反转
    - 状态持续时间 > 60（状态老化）

    优点：在状态早期入场，捕捉新趋势启动
    缺点：频繁的短暂状态切换会产生噪音交易
    """
    name = "v68_regime_duration_supertrend"
    warmup = 250
    freq = "4h"

    regime_period: int = 60
    fresh_threshold: int = 20
    stale_threshold: int = 60
    st_period: int = 10
    st_mult: float = 3.0
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._regime = None
        self._cur_duration = None
        self._st_dir = None
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

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._regime, _, _ = vol_regime_simple(closes, period=self.regime_period)
        _, _, self._cur_duration = switch_speed(self._regime, period=self.regime_period)
        _, self._st_dir = supertrend(highs, lows, closes,
                                      period=self.st_period, multiplier=self.st_mult)
        self._atr = atr(highs, lows, closes, period=self.atr_period)

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        cur_dur = self._cur_duration[i]
        st_dir = self._st_dir[i]
        atr_val = self._atr[i]
        if np.isnan(cur_dur) or np.isnan(st_dir) or np.isnan(atr_val):
            return

        is_fresh = cur_dur <= self.fresh_threshold
        is_stale = cur_dur > self.stale_threshold
        self.bars_since_last_scale += 1

        # ── 1. 止损检查 ──
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

        # ── 3. 信号弱化退出 ──
        if side == 1 and (st_dir != 1 or is_stale):
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and (st_dir != -1 or is_stale):
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场逻辑 ──
        if side == 0 and is_fresh:
            if st_dir == 1:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self._set_entry(price, price - self.atr_stop_mult * atr_val)
            elif st_dir == -1:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self._set_entry(price, price + self.atr_stop_mult * atr_val)

        # ── 5. 加仓逻辑 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and st_dir == 1 and not is_stale):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and st_dir == -1 and not is_stale):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.sell(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _set_entry(self, price, stop):
        self.entry_price = price
        self.stop_price = stop
        self.highest_since_entry = price
        self.lowest_since_entry = price
        self.position_scale = 1
        self.bars_since_last_scale = 0

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
