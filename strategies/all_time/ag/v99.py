import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.hma import hma
from indicators.volatility.bollinger import bollinger_bands
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class HMABollingerWidthStrategy(TimeSeriesStrategy):
    """
    策略简介：HMA 方向变化 + Bollinger Width 低位时入场的突破策略。

    使用指标：
    - HMA(20): Hull Moving Average，低滞后均线，方向变化检测趋势转换
    - Bollinger Bands(20,2.0): 带宽（width）处于低位=波动压缩，突破预兆
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - HMA 方向从下转上（斜率从负转正）
    - BB width < 历史20周期均值（波动率压缩状态）

    进场条件（做空）：
    - HMA 方向从上转下（斜率从正转负）
    - BB width < 历史20周期均值（波动率压缩状态）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - HMA 方向再次反转

    优点：HMA低滞后+BB width压缩=高质量方向突破信号
    缺点：BB width周期性可能导致长期等待
    """
    name = "v99_hma_bbwidth"
    warmup = 200
    freq = "daily"

    hma_period: int = 20
    bb_period: int = 20
    bb_std: float = 2.0
    width_lookback: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._hma = None
        self._bb_upper = None
        self._bb_middle = None
        self._bb_lower = None
        self._bb_width = None
        self._bb_width_avg = None
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

        self._hma = hma(closes, self.hma_period)
        self._bb_upper, self._bb_middle, self._bb_lower = bollinger_bands(
            closes, self.bb_period, self.bb_std)

        # BB width = (upper - lower) / middle
        n = len(closes)
        self._bb_width = np.full(n, np.nan)
        for idx in range(n):
            if (not np.isnan(self._bb_upper[idx]) and
                    not np.isnan(self._bb_lower[idx]) and
                    not np.isnan(self._bb_middle[idx]) and
                    self._bb_middle[idx] > 0):
                self._bb_width[idx] = (self._bb_upper[idx] - self._bb_lower[idx]) / self._bb_middle[idx]

        # Rolling average of BB width
        self._bb_width_avg = np.full(n, np.nan)
        for idx in range(self.width_lookback, n):
            window = self._bb_width[idx - self.width_lookback:idx]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                self._bb_width_avg[idx] = np.mean(valid)

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

        hma_val = self._hma[i]
        bb_w = self._bb_width[i]
        bb_w_avg = self._bb_width_avg[i]
        atr_val = self._atr[i]
        if np.isnan(hma_val) or np.isnan(bb_w) or np.isnan(bb_w_avg) or np.isnan(atr_val):
            return

        if i < 2:
            return
        prev_hma = self._hma[i - 1]
        prev_prev_hma = self._hma[i - 2]
        if np.isnan(prev_hma) or np.isnan(prev_prev_hma):
            return

        cur_slope = hma_val - prev_hma
        prev_slope = prev_hma - prev_prev_hma
        hma_turn_up = prev_slope <= 0 and cur_slope > 0
        hma_turn_down = prev_slope >= 0 and cur_slope < 0
        low_width = bb_w < bb_w_avg

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
        if side == 1 and cur_slope < 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and cur_slope > 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0 and low_width:
            if hma_turn_up:
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
            elif hma_turn_down:
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
                    and cur_slope > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and cur_slope < 0):
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
