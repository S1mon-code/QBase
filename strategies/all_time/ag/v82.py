import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.momentum.tsi import tsi
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class TSISupertrendStrategy(TimeSeriesStrategy):
    """
    策略简介：TSI 零轴交叉 + Supertrend 趋势方向一致时入场的多空策略。

    使用指标：
    - TSI(25,13,7): True Strength Index，零轴交叉判断动量转换
    - Supertrend(10,3.0): 趋势方向过滤，只在趋势一致时交易
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - TSI 线上穿零轴（动量转正）
    - Supertrend 方向 = 1（上升趋势）

    进场条件（做空）：
    - TSI 线下穿零轴（动量转负）
    - Supertrend 方向 = -1（下降趋势）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Supertrend 反转或 TSI 回穿零轴

    优点：TSI双重平滑减少噪音，Supertrend过滤假突破
    缺点：TSI滞后较大，快速反转时可能错过最佳入场
    """
    name = "v82_tsi_supertrend"
    warmup = 500
    freq = "1h"

    tsi_long: int = 25
    tsi_short: int = 13
    tsi_signal: int = 7
    st_period: int = 10
    st_mult: float = 3.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._tsi_line = None
        self._tsi_signal = None
        self._st_line = None
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
        self.direction = 0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._tsi_line, self._tsi_signal = tsi(
            closes, self.tsi_long, self.tsi_short, self.tsi_signal)
        self._st_line, self._st_dir = supertrend(
            highs, lows, closes, self.st_period, self.st_mult)
        self._atr = atr(highs, lows, closes, period=14)

        window = 20
        self._avg_volume = np.full_like(volumes, np.nan)
        for idx in range(window, len(volumes)):
            self._avg_volume[idx] = np.mean(volumes[idx - window:idx])

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        tsi_val = self._tsi_line[i]
        st_dir = self._st_dir[i]
        atr_val = self._atr[i]
        if np.isnan(tsi_val) or np.isnan(atr_val) or np.isnan(st_dir):
            return

        self.bars_since_last_scale += 1

        prev_tsi = self._tsi_line[i - 1] if i > 0 else np.nan
        if np.isnan(prev_tsi):
            return

        tsi_cross_up = prev_tsi <= 0 and tsi_val > 0
        tsi_cross_down = prev_tsi >= 0 and tsi_val < 0

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

        # ── 3. 信号反转退出 ──
        if side == 1 and (st_dir == -1 or tsi_cross_down):
            context.close_long()
            self._reset_state()
        elif side == -1 and (st_dir == 1 or tsi_cross_up):
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if tsi_cross_up and st_dir == 1:
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
            elif tsi_cross_down and st_dir == -1:
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

        # ── 5. 加仓逻辑 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and st_dir == 1 and tsi_val > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and st_dir == -1 and tsi_val < 0):
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
