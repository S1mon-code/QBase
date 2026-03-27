import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.momentum.macd import macd
from indicators.volume.obv import obv
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class MACDOBVStrategy(TimeSeriesStrategy):
    """
    策略简介：MACD 动量交叉 + OBV 成交量趋势确认的多空策略。

    使用指标：
    - MACD(12,26,9): 动量方向判断，MACD线穿越信号线时产生信号
    - OBV: 成交量累积方向确认，用EMA平滑后判断趋势
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - MACD 线上穿信号线（金叉）
    - OBV 短期EMA > OBV 长期EMA（量能向上）

    进场条件（做空）：
    - MACD 线下穿信号线（死叉）
    - OBV 短期EMA < OBV 长期EMA（量能向下）

    出场条件：
    - ATR 追踪止损（最高/低价回撤 N×ATR）
    - 分层止盈（3ATR 减 1/3，5ATR 再减 1/3）
    - 信号反转（MACD交叉方向改变）

    优点：MACD趋势确认+OBV量能验证，双重过滤减少假信号
    缺点：MACD本身滞后，震荡市中频繁交叉导致whipsaw
    """
    name = "v81_macd_obv"
    warmup = 250
    freq = "4h"

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    obv_fast: int = 10
    obv_slow: int = 30
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._macd_line = None
        self._macd_signal = None
        self._macd_hist = None
        self._obv_fast_ema = None
        self._obv_slow_ema = None
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
        self.direction = 0  # 1=long, -1=short

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._macd_line, self._macd_signal, self._macd_hist = macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal)
        self._atr = atr(highs, lows, closes, period=14)

        obv_arr = obv(closes, volumes)
        # EMA smooth OBV
        from indicators.trend.ema import ema
        self._obv_fast_ema = ema(obv_arr, self.obv_fast)
        self._obv_slow_ema = ema(obv_arr, self.obv_slow)

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

        macd_val = self._macd_line[i]
        sig_val = self._macd_signal[i]
        atr_val = self._atr[i]
        obv_f = self._obv_fast_ema[i]
        obv_s = self._obv_slow_ema[i]
        if np.isnan(macd_val) or np.isnan(sig_val) or np.isnan(atr_val):
            return
        if np.isnan(obv_f) or np.isnan(obv_s):
            return

        self.bars_since_last_scale += 1

        prev_macd = self._macd_line[i - 1] if i > 0 else np.nan
        prev_sig = self._macd_signal[i - 1] if i > 0 else np.nan
        if np.isnan(prev_macd) or np.isnan(prev_sig):
            return

        macd_cross_up = prev_macd <= prev_sig and macd_val > sig_val
        macd_cross_down = prev_macd >= prev_sig and macd_val < sig_val
        obv_bullish = obv_f > obv_s
        obv_bearish = obv_f < obv_s

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
        if side == 1 and macd_cross_down:
            context.close_long()
            self._reset_state()
            # fall through to check short entry
        elif side == -1 and macd_cross_up:
            context.close_short()
            self._reset_state()
            # fall through to check long entry

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if macd_cross_up and obv_bullish:
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
            elif macd_cross_down and obv_bearish:
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
                    and obv_bullish and macd_val > sig_val):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and obv_bearish and macd_val < sig_val):
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
