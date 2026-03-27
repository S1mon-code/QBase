import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.wavelet_decompose import wavelet_features
from indicators.regime.market_state import market_state
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class WaveletMarketPhaseATRBandsStrategy(TimeSeriesStrategy):
    """
    策略简介：Wavelet 分解提取趋势 + 市场阶段（accumulation/distribution）识别 + ATR 动态止损的日频多空策略。

    使用指标：
    - Wavelet Features(level=4): 小波分解分离趋势与噪声，trend_component 为低频方向
    - Market State(20): 多维度市场状态分类（0=安静,1=上涨趋势,2=下跌趋势,3=高波动震荡,4=突破）
    - ATR(14): 止损距离计算 + 动态 ATR band 过滤

    进场条件（做多）：
    - Wavelet trend 斜率 > 0（低频趋势向上）
    - Market state == 1（上涨趋势）或 4（向上突破）
    - Price > wavelet trend（价格在趋势之上）

    进场条件（做空）：
    - Wavelet trend 斜率 < 0
    - Market state == 2（下跌趋势）或 4（向下突破）
    - Price < wavelet trend

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Market state 变为 0（安静）或 3（高波动震荡）

    优点：小波分解有效去噪 + 市场阶段识别提供结构性信号
    缺点：小波分解在端点有边缘效应，市场状态分类依赖 OI 数据
    """
    name = "v144_wavelet_market_phase_atr_bands"
    warmup = 250
    freq = "daily"

    wavelet_level: int = 4
    ms_period: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._trend = None
        self._detail = None
        self._state = None
        self._state_conf = None
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

        self._trend, self._detail, _ = wavelet_features(closes, level=self.wavelet_level)
        self._state, self._state_conf = market_state(closes, volumes, oi, period=self.ms_period)
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

        trend_val = self._trend[i]
        state_val = self._state[i]
        atr_val = self._atr[i]
        if np.isnan(trend_val) or np.isnan(atr_val):
            return
        if np.isnan(state_val):
            state_val = 0.0

        # Compute trend slope (difference of last 2 bars)
        trend_slope = 0.0
        if i >= 1 and not np.isnan(self._trend[i - 1]):
            trend_slope = trend_val - self._trend[i - 1]

        self.bars_since_last_scale += 1
        state_int = int(state_val)

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

        # ── 3. 状态退出：安静或高波动震荡 ──
        if side == 1 and state_int in (0, 3):
            context.close_long()
            self._reset_state()
        elif side == -1 and state_int in (0, 3):
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if trend_slope > 0 and price > trend_val and state_int in (1, 4):
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
            elif trend_slope < 0 and price < trend_val and state_int in (2, 4):
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
                    and trend_slope > 0 and state_int == 1):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and trend_slope < 0 and state_int == 2):
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
