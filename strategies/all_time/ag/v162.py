import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.wavelet_decompose import wavelet_features
from indicators.momentum.cci import cci
from indicators.volume.klinger import klinger
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class WaveletCCIKlingerStrategy(TimeSeriesStrategy):
    """
    策略简介：Wavelet趋势分解 + CCI突破 + Klinger量能确认的4h多空策略。

    使用指标：
    - Wavelet(level=4): 分离趋势与噪音，趋势方向用于方向判断
    - CCI(20): 动量突破信号，>100做多/<-100做空
    - Klinger(34,55,13): 量能方向确认，KVO>signal=多头量能
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Wavelet趋势上升（当前趋势 > 前一趋势）
    - CCI > 100（动量突破上轨）
    - KVO > signal（量能确认上涨）

    进场条件（做空）：
    - Wavelet趋势下降（当前趋势 < 前一趋势）
    - CCI < -100（动量突破下轨）
    - KVO < signal（量能确认下跌）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - CCI回归零轴

    优点：降噪后趋势更清晰+CCI提供明确突破点+量能三重确认
    缺点：Wavelet在趋势转折时滞后，CCI突破在震荡市假信号多
    """
    name = "v162_wavelet_cci_klinger"
    warmup = 400
    freq = "4h"

    cci_period: int = 20
    cci_thresh: float = 100.0
    klinger_fast: int = 34
    klinger_slow: int = 55
    klinger_signal: int = 13
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._trend = None
        self._cci = None
        self._kvo = None
        self._kvo_sig = None
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

        self._trend, _, _ = wavelet_features(closes, level=4)
        self._cci = cci(highs, lows, closes, period=self.cci_period)
        self._kvo, self._kvo_sig = klinger(highs, lows, closes, volumes,
                                            self.klinger_fast, self.klinger_slow,
                                            self.klinger_signal)
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

        if i < 1:
            return
        trend_now = self._trend[i]
        trend_prev = self._trend[i - 1]
        cci_val = self._cci[i]
        kvo_val = self._kvo[i]
        kvo_sig = self._kvo_sig[i]
        atr_val = self._atr[i]
        if np.isnan(trend_now) or np.isnan(trend_prev) or np.isnan(cci_val) or np.isnan(atr_val):
            return
        if np.isnan(kvo_val) or np.isnan(kvo_sig):
            return

        self.bars_since_last_scale += 1
        trend_up = trend_now > trend_prev
        trend_down = trend_now < trend_prev

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

        # ── 3. CCI回归零轴退出 ──
        if side == 1 and cci_val < 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and cci_val > 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if trend_up and cci_val > self.cci_thresh and kvo_val > kvo_sig:
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
            elif trend_down and cci_val < -self.cci_thresh and kvo_val < kvo_sig:
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
                    and trend_up and cci_val > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and trend_down and cci_val < 0):
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
