import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.bayesian_trend import bayesian_online_trend
from indicators.volatility.atr import atr
from indicators.volume.mfi import mfi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BayesianChangepointATRMFIStrategy(TimeSeriesStrategy):
    """
    策略简介：Bayesian Changepoint检测 + ATR扩张 + MFI方向确认的1h多空策略。

    使用指标：
    - Bayesian Online Trend(0.01): 变点检测+趋势斜率估计
    - ATR(14): 波动率扩张检测+止损距离
    - MFI(14): 资金流向指数，>50=资金流入，<50=资金流出

    进场条件（做多）：
    - Bayesian趋势斜率 > 0（检测到上升趋势变点）
    - ATR > 1.5倍20期均值（波动率正在扩张）
    - MFI > 50（资金流入确认）

    进场条件（做空）：
    - Bayesian趋势斜率 < 0（检测到下降趋势变点）
    - ATR > 1.5倍20期均值（波动率正在扩张）
    - MFI < 50（资金流出确认）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - MFI方向反转

    优点：变点检测能快速识别趋势转换+ATR扩张过滤低波动假信号
    缺点：1h频率噪音大，变点检测可能过度敏感
    """
    name = "v167_bayesian_atr_mfi"
    warmup = 800
    freq = "1h"

    hazard_rate: float = 0.01
    atr_expansion: float = 1.5
    mfi_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._bay_trend = None
        self._atr = None
        self._atr_ma = None
        self._mfi = None
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

        self._bay_trend = bayesian_online_trend(closes, hazard_rate=self.hazard_rate)
        self._atr = atr(highs, lows, closes, period=14)
        self._mfi = mfi(highs, lows, closes, volumes, period=self.mfi_period)

        # Rolling ATR mean for expansion detection
        n = len(closes)
        self._atr_ma = np.full(n, np.nan)
        window = 20
        for idx in range(window, n):
            vals = self._atr[idx-window:idx]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                self._atr_ma[idx] = np.mean(vals)

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

        bay_val = self._bay_trend[i]
        atr_val = self._atr[i]
        atr_ma = self._atr_ma[i]
        mfi_val = self._mfi[i]
        if np.isnan(bay_val) or np.isnan(atr_val) or np.isnan(atr_ma) or np.isnan(mfi_val):
            return

        self.bars_since_last_scale += 1
        atr_expanding = atr_val > atr_ma * self.atr_expansion

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

        # ── 3. MFI方向反转退出 ──
        if side == 1 and mfi_val < 40:
            context.close_long()
            self._reset_state()
        elif side == -1 and mfi_val > 60:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0 and atr_expanding:
            if bay_val > 0 and mfi_val > 50:
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
            elif bay_val < 0 and mfi_val < 50:
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
                    and bay_val > 0 and mfi_val > 50):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and bay_val < 0 and mfi_val < 50):
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
