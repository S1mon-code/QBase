import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.trend_filter import l1_trend_filter
from indicators.microstructure.price_efficiency import price_efficiency_coefficient
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class DenoisingPriceEfficiencyStrategy(TimeSeriesStrategy):
    """
    策略简介：L1趋势滤波去噪方向 + 价格效率低时入场的多空策略。

    使用指标：
    - L1 Trend Filter (Denoising): 分段线性趋势提取，去除高频噪音
    - Price Efficiency Coefficient: PEC接近0=噪音大（均值回归机会），
      PEC接近1=高效趋势（趋势跟随机会）
    - ATR(14): 止损距离计算

    策略逻辑：
    - PEC < 0.3（低效率=价格混沌）: 交易去噪后的趋势方向（噪音中找信号）
    - PEC > 0.7（高效率=清晰趋势）: 不入场（已经price-in）

    进场条件（做多）：
    - L1趋势斜率 > 0（去噪后方向向上）
    - PEC < 0.3（市场低效率，去噪信号有alpha）

    进场条件（做空）：
    - L1趋势斜率 < 0（去噪后方向向下）
    - PEC < 0.3

    出场条件：
    - ATR追踪止损 / 分层止盈 / 趋势斜率反转

    优点：L1滤波保留趋势断点（比EMA/Kalman更适合段式行情），PEC过滤高效市场
    缺点：L1计算较慢，PEC阈值对品种敏感
    """
    name = "v59_denoising_price_efficiency"
    warmup = 500
    freq = "4h"

    l1_lambda: float = 1.0
    pec_period: int = 20
    pec_threshold: float = 0.3
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._trend = None
        self._trend_slope = None
        self._pec = None
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
        n = len(closes)

        self._trend, _, _ = l1_trend_filter(closes, lambda_val=self.l1_lambda)

        # Trend slope: 5-bar difference of denoised trend
        self._trend_slope = np.full(n, np.nan)
        for idx in range(5, n):
            if not np.isnan(self._trend[idx]) and not np.isnan(self._trend[idx - 5]):
                self._trend_slope[idx] = self._trend[idx] - self._trend[idx - 5]

        self._pec, _ = price_efficiency_coefficient(closes, period=self.pec_period)
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

        trend_slope = self._trend_slope[i]
        pec_val = self._pec[i]
        atr_val = self._atr[i]
        if np.isnan(trend_slope) or np.isnan(pec_val) or np.isnan(atr_val) or atr_val <= 0:
            return

        self.bars_since_last_scale += 1

        # ── 1. 止损 ──
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
        if side == 1 and trend_slope < 0:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and trend_slope > 0:
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场 (only when price efficiency is low) ──
        if side == 0 and pec_val < self.pec_threshold:
            if trend_slope > 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
            elif trend_slope < 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0

        # ── 5. 加仓 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and trend_slope > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and trend_slope < 0):
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
