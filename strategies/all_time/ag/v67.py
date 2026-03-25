import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.regime.variance_ratio import variance_ratio_test
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class VarianceRatioROC(TimeSeriesStrategy):
    """
    策略简介：方差比检测市场效率 + ROC 动量。VR>1 动量跟随，VR<1 均值回归。

    使用指标：
    - variance_ratio_test(60, 5): 方差比 (VR>1=趋势, VR<1=均值回归)
    - rate_of_change(20): 价格变化率
    - ATR(14): 止损距离计算

    进场条件（动量跟随，VR > 1.1）：
    - VR > 1.1（确认趋势环境）
    - ROC > roc_threshold 做多 / ROC < -roc_threshold 做空

    进场条件（均值回归，VR < 0.9）：
    - VR < 0.9（确认均值回归环境）
    - ROC > roc_threshold 做空 / ROC < -roc_threshold 做多（反向）

    出场条件：
    - ATR 追踪止损 / 分层止盈
    - VR 回到中性区域 (0.9~1.1)

    优点：自适应趋势和均值回归两种市场状态
    缺点：方差比临界值附近频繁切换可能导致震荡
    """
    name = "v67_variance_ratio_roc"
    warmup = 250
    freq = "4h"

    vr_period: int = 60
    vr_holding: int = 5
    vr_trend_threshold: float = 1.1
    vr_mr_threshold: float = 0.9
    roc_period: int = 20
    roc_threshold: float = 2.0
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._vr = None
        self._roc = None
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
        self.entry_mode = 0  # 1=momentum, -1=mean_reversion

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._vr, _ = variance_ratio_test(closes, period=self.vr_period,
                                           holding=self.vr_holding)
        self._roc = rate_of_change(closes, period=self.roc_period)
        self._atr = atr(highs, lows, closes, period=self.atr_period)

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

        vr_val = self._vr[i]
        roc_val = self._roc[i]
        atr_val = self._atr[i]
        if np.isnan(vr_val) or np.isnan(roc_val) or np.isnan(atr_val):
            return

        is_trending = vr_val > self.vr_trend_threshold
        is_mr = vr_val < self.vr_mr_threshold
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
        if side != 0:
            # 动量模式下 VR 回到中性，或反转
            if self.entry_mode == 1 and not is_trending:
                if side == 1:
                    context.close_long()
                else:
                    context.close_short()
                self._reset_state()
                return
            # 均值回归模式下 VR 不再是 MR
            if self.entry_mode == -1 and not is_mr:
                if side == 1:
                    context.close_long()
                else:
                    context.close_short()
                self._reset_state()
                return

        # ── 4. 入场逻辑 ──
        if side == 0:
            if is_trending:
                if roc_val > self.roc_threshold:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.buy(base_lots)
                        self._set_entry(price, price - self.atr_stop_mult * atr_val, 1)
                elif roc_val < -self.roc_threshold:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.sell(base_lots)
                        self._set_entry(price, price + self.atr_stop_mult * atr_val, 1)
            elif is_mr:
                if roc_val < -self.roc_threshold:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.buy(base_lots)
                        self._set_entry(price, price - self.atr_stop_mult * atr_val, -1)
                elif roc_val > self.roc_threshold:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.sell(base_lots)
                        self._set_entry(price, price + self.atr_stop_mult * atr_val, -1)

        # ── 5. 加仓逻辑 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.sell(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _set_entry(self, price, stop, mode):
        self.entry_price = price
        self.stop_price = stop
        self.highest_since_entry = price
        self.lowest_since_entry = price
        self.position_scale = 1
        self.bars_since_last_scale = 0
        self.entry_mode = mode

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
        self.entry_mode = 0
