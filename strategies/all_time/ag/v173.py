import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.wavelet_decompose import wavelet_features
from indicators.trend.psar import psar
from indicators.structure.squeeze_detector import squeeze_probability
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class DenoisingPSARSqueezeStrategy(TimeSeriesStrategy):
    """
    策略简介：Wavelet降噪 + PSAR趋势翻转 + Squeeze释放的4h多空策略。

    使用指标：
    - Wavelet(level=4): 降噪获取干净趋势方向
    - PSAR(0.02,0.02,0.2): Parabolic SAR翻转信号
    - Squeeze Detector(20): OI+价格+量检测squeeze条件
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Wavelet趋势上升
    - PSAR翻转为多头（dir从-1变1）
    - Squeeze概率下降（squeeze释放，非squeeze释放也可在趋势中入场）

    进场条件（做空）：
    - Wavelet趋势下降
    - PSAR翻转为空头（dir从1变-1）
    - Squeeze概率下降

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - PSAR再次翻转

    优点：降噪减少PSAR假翻转+squeeze释放提供高质量入场点
    缺点：Wavelet滞后可能错过转折初期，squeeze信号稀疏
    """
    name = "v173_denoising_psar_squeeze"
    warmup = 500
    freq = "4h"

    af_start: float = 0.02
    af_step: float = 0.02
    af_max: float = 0.2
    squeeze_period: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._trend = None
        self._psar_dir = None
        self._squeeze_prob = None
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
        oi = context.get_full_oi_array()

        self._atr = atr(highs, lows, closes, period=14)
        self._trend, _, _ = wavelet_features(closes, level=4)
        _, self._psar_dir = psar(highs, lows, self.af_start, self.af_step, self.af_max)
        self._squeeze_prob, _ = squeeze_probability(closes, oi, volumes, period=self.squeeze_period)

        window = 20
        self._avg_volume = np.full_like(volumes, np.nan)
        for idx in range(window, len(volumes)):
            self._avg_volume[idx] = np.mean(volumes[idx-window:idx])

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        if i < 2:
            return
        trend_now = self._trend[i]
        trend_prev = self._trend[i - 1]
        psar_dir = self._psar_dir[i]
        psar_prev = self._psar_dir[i - 1]
        sq_prob = self._squeeze_prob[i]
        atr_val = self._atr[i]
        if np.isnan(trend_now) or np.isnan(trend_prev) or np.isnan(psar_dir) or np.isnan(atr_val):
            return
        if np.isnan(psar_prev):
            return
        if np.isnan(sq_prob):
            sq_prob = 0.0

        self.bars_since_last_scale += 1
        trend_up = trend_now > trend_prev
        trend_down = trend_now < trend_prev
        psar_flip_up = psar_prev == -1 and psar_dir == 1
        psar_flip_down = psar_prev == 1 and psar_dir == -1

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

        # ── 3. PSAR翻转退出 ──
        if side == 1 and psar_flip_down:
            context.close_long()
            self._reset_state()
        elif side == -1 and psar_flip_up:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if trend_up and psar_flip_up and sq_prob < 0.5:
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
            elif trend_down and psar_flip_down and sq_prob < 0.5:
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
                    and trend_up and psar_dir == 1):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and trend_down and psar_dir == -1):
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
