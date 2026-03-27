import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.regime.fractal_dimension import fractal_dim
from indicators.momentum.macd import macd
from indicators.volume.obv import obv
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class FractalDimMACDOBVStrategy(TimeSeriesStrategy):
    """
    策略简介：Fractal Dimension低值 + MACD交叉 + OBV确认的4h多空策略。

    使用指标：
    - Fractal Dimension(60): 低值(<1.5)=趋势市，高值=随机/震荡
    - MACD(12,26,9): 金叉/死叉信号
    - OBV: 量价趋势方向确认
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Fractal Dim < 1.5（趋势性市场）
    - MACD线 > Signal线（金叉）且MACD > 0
    - OBV上升（20bar对比）

    进场条件（做空）：
    - Fractal Dim < 1.5（趋势性市场）
    - MACD线 < Signal线（死叉）且MACD < 0
    - OBV下降

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - MACD交叉反转

    优点：Fractal dimension科学筛选趋势市+MACD经典动量+OBV量确认
    缺点：Fractal dimension计算窗口大导致滞后，MACD在趋势末段假信号
    """
    name = "v176_fractal_dim_macd_obv"
    warmup = 500
    freq = "4h"

    fd_period: int = 60
    fd_thresh: float = 1.5
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    obv_lookback: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._fd = None
        self._macd_line = None
        self._macd_sig = None
        self._macd_hist = None
        self._obv = None
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

        self._atr = atr(highs, lows, closes, period=14)
        self._fd = fractal_dim(closes, period=self.fd_period)
        self._macd_line, self._macd_sig, self._macd_hist = macd(
            closes, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        self._obv = obv(closes, volumes)

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

        if i < self.obv_lookback + 1:
            return
        fd_val = self._fd[i]
        ml = self._macd_line[i]
        ms = self._macd_sig[i]
        obv_now = self._obv[i]
        obv_prev = self._obv[i - self.obv_lookback]
        atr_val = self._atr[i]
        if np.isnan(fd_val) or np.isnan(ml) or np.isnan(ms) or np.isnan(atr_val):
            return
        if np.isnan(obv_now) or np.isnan(obv_prev):
            return

        self.bars_since_last_scale += 1
        trending = fd_val < self.fd_thresh
        obv_up = obv_now > obv_prev
        obv_down = obv_now < obv_prev

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

        # ── 3. MACD反转退出 ──
        if side == 1 and ml < ms:
            context.close_long()
            self._reset_state()
        elif side == -1 and ml > ms:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0 and trending:
            if ml > ms and ml > 0 and obv_up:
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
            elif ml < ms and ml < 0 and obv_down:
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
                    and ml > ms and obv_up):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and ml < ms and obv_down):
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
