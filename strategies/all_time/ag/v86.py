import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.momentum.kama import kama
from indicators.microstructure.amihud import amihud_illiquidity
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class KAMAAmihudStrategy(TimeSeriesStrategy):
    """
    策略简介：KAMA 自适应均线方向 + Amihud 流动性过滤的多空策略。

    使用指标：
    - KAMA(10,2,30): Kaufman自适应均线，高效率时紧跟价格
    - Amihud Illiquidity(20): 流动性衡量，高流动性时跟随KAMA，低流动性时fade
    - ATR(14): 止损距离计算

    进场条件（做多 - 高流动性）：
    - 价格 > KAMA 且 KAMA 斜率上升
    - Amihud z-score < 1.0（流动性正常/良好）

    进场条件（做空 - 高流动性）：
    - 价格 < KAMA 且 KAMA 斜率下降
    - Amihud z-score < 1.0（流动性正常/良好）

    逆向模式（低流动性 fade）：
    - Amihud z-score > 2.0 时，价格偏离KAMA过远则fade回均值

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - KAMA 斜率反转

    优点：自适应+流动性过滤，在不同市场微观结构下灵活切换
    缺点：Amihud在期货中可能不如股票有效，参数较多
    """
    name = "v86_kama_amihud"
    warmup = 500
    freq = "1h"

    kama_period: int = 10
    kama_fast: int = 2
    kama_slow: int = 30
    amihud_period: int = 20
    illiquidity_threshold: float = 1.0
    fade_threshold: float = 2.0
    kama_slope_len: int = 5
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._kama = None
        self._amihud_raw = None
        self._amihud_z = None
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

        self._kama = kama(closes, self.kama_period, self.kama_fast, self.kama_slow)
        self._amihud_raw, self._amihud_z = amihud_illiquidity(
            closes, volumes, self.amihud_period)
        self._atr = atr(highs, lows, closes, period=14)

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

        kama_val = self._kama[i]
        amihud_z = self._amihud_z[i]
        atr_val = self._atr[i]
        if np.isnan(kama_val) or np.isnan(amihud_z) or np.isnan(atr_val):
            return

        # KAMA slope
        if i < self.kama_slope_len:
            return
        prev_kama = self._kama[i - self.kama_slope_len]
        if np.isnan(prev_kama):
            return
        kama_slope = kama_val - prev_kama

        self.bars_since_last_scale += 1

        is_liquid = amihud_z < self.illiquidity_threshold
        is_illiquid = amihud_z > self.fade_threshold

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
        if side == 1 and kama_slope < 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and kama_slope > 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if is_liquid:
                # 趋势跟随模式
                if price > kama_val and kama_slope > 0:
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
                elif price < kama_val and kama_slope < 0:
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
            elif is_illiquid:
                # 均值回归模式：价格远离KAMA时fade
                deviation = (price - kama_val) / atr_val
                if deviation > 2.0:
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
                elif deviation < -2.0:
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

        # ── 5. 加仓逻辑 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and kama_slope > 0 and is_liquid):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and kama_slope < 0 and is_liquid):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.sell(add)
                self.position_scale += 1
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
        self.direction = 0
