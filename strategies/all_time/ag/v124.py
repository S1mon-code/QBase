import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.wavelet_decompose import wavelet_features
from indicators.regime.spectral_density import dominant_cycle
from indicators.seasonality.seasonal_momentum import seasonal_momentum

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV124(TimeSeriesStrategy):
    """
    策略简介：Wavelet 分解方向 + 周期相位识别 + 季节性动量对齐的多空策略。

    使用指标：
    - Wavelet Features(db4, level=4): 多尺度分解提取趋势方向
    - Dominant Cycle(120): 识别当前主导周期相位
    - Seasonal Momentum(3yr): 历史同期季节性动量

    进场条件（做多）：Wavelet 趋势分量上升，周期相位上升，季节性动量>0
    进场条件（做空）：Wavelet 趋势分量下降，周期相位下降，季节性动量<0

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Wavelet 趋势反转

    优点：多尺度分析过滤噪音，季节性提供额外确认维度
    缺点：Wavelet 在序列边缘有边界效应，季节性可能失效
    """
    name = "ag_alltime_v124"
    warmup = 500
    freq = "daily"

    wavelet_name: str = "db4"
    wavelet_level: int = 4
    cycle_period: int = 120
    seasonal_years: int = 3
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._wavelet = None
        self._cycle = None
        self._seasonal = None
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

        # Wavelet: returns matrix of decomposed components
        self._wavelet = wavelet_features(closes, wavelet=self.wavelet_name, level=self.wavelet_level)

        self._cycle = dominant_cycle(closes, period=self.cycle_period)

        datetimes = context.get_full_datetime_array()
        self._seasonal = seasonal_momentum(closes, datetimes, lookback_years=self.seasonal_years)

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        # Wavelet trend: use first (lowest freq) component direction
        if self._wavelet.ndim == 2:
            wv_val = self._wavelet[i, 0]
            wv_prev = self._wavelet[i - 1, 0] if i > 0 else np.nan
        else:
            wv_val = self._wavelet[i]
            wv_prev = self._wavelet[i - 1] if i > 0 else np.nan
        cycle_val = self._cycle[i]
        seas_val = self._seasonal[i]
        if np.isnan(wv_val) or np.isnan(wv_prev) or np.isnan(cycle_val) or np.isnan(seas_val):
            return

        wv_up = wv_val > wv_prev
        wv_down = wv_val < wv_prev

        # Cycle phase: positive = upswing, negative = downswing
        cycle_up = cycle_val > 0
        cycle_down = cycle_val < 0

        self.bars_since_last_scale += 1

        # 1. Stop loss
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

        # 2. Tiered profit-taking
        if side != 0 and self.entry_price > 0:
            if side == 1:
                profit_atr = (price - self.entry_price) / atr_val
            else:
                profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_3atr = True
                return

        # 3. Signal-based exit
        if side == 1 and wv_down:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and wv_up:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if wv_up and cycle_up and seas_val > 0:
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
            elif wv_down and cycle_down and seas_val < 0:
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

        # 5. Scale-in
        elif side != 0 and self._should_add(price, atr_val):
            if self.direction == 1 and wv_up:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and wv_down:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.sell(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1 and price < self.entry_price + atr_val:
            return False
        if self.direction == -1 and price > self.entry_price - atr_val:
            return False
        return True

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

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
