import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.bayesian_trend import bayesian_online_trend
from indicators.volatility.bollinger import bollinger_bands
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BayesianBBSqueezeStrategy(TimeSeriesStrategy):
    """
    策略简介：贝叶斯变点检测 + 布林带收缩入场的多空策略。

    使用指标：
    - Bayesian Online Trend: 变点概率 + 趋势方向，检测市场结构变化
    - Bollinger Bands: 带宽收缩（squeeze）表示低波动蓄势
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - BB带宽处于低位（squeeze状态：带宽 < 20日均带宽的0.8倍）
    - 贝叶斯趋势斜率 > 0（变点后趋势向上）
    - 变点概率近期触发过（5bar内changepoint_prob > 0.3）

    进场条件（做空）：
    - BB squeeze状态
    - 贝叶斯趋势斜率 < 0（变点后趋势向下）
    - 变点概率近期触发过

    出场条件：
    - ATR追踪止损 / 分层止盈 / 趋势斜率反转

    优点：变点检测捕捉市场结构转折，BB squeeze精确定位低波动蓄势
    缺点：变点检测可能滞后，squeeze后不一定有方向性突破
    """
    name = "v50_bayesian_bb_squeeze"
    warmup = 600
    freq = "1h"

    hazard_rate: float = 0.01
    bb_period: int = 20
    bb_std: float = 2.0
    squeeze_ratio: float = 0.8
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._trend_est = None
        self._cp_prob = None
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

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        n = len(closes)

        _, self._cp_prob, self._trend_est = bayesian_online_trend(
            closes, hazard_rate=self.hazard_rate)

        upper, middle, lower = bollinger_bands(closes, self.bb_period, self.bb_std)
        self._bb_width = np.full(n, np.nan)
        for idx in range(n):
            if not np.isnan(middle[idx]) and middle[idx] > 0:
                self._bb_width[idx] = (upper[idx] - lower[idx]) / middle[idx]

        # Rolling average of BB width
        self._bb_width_avg = np.full(n, np.nan)
        for idx in range(self.bb_period, n):
            w = self._bb_width[idx - self.bb_period:idx]
            valid = w[~np.isnan(w)]
            if len(valid) > 0:
                self._bb_width_avg[idx] = np.mean(valid)

        self._atr = atr(highs, lows, closes, period=self.atr_period)

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def _recent_changepoint(self, i, lookback=5):
        """Check if changepoint probability was high in recent bars."""
        for j in range(max(0, i - lookback), i + 1):
            if not np.isnan(self._cp_prob[j]) and self._cp_prob[j] > 0.3:
                return True
        return False

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        trend_val = self._trend_est[i]
        bb_w = self._bb_width[i]
        bb_w_avg = self._bb_width_avg[i]
        atr_val = self._atr[i]
        if np.isnan(trend_val) or np.isnan(bb_w) or np.isnan(bb_w_avg) or np.isnan(atr_val) or atr_val <= 0:
            return

        self.bars_since_last_scale += 1
        in_squeeze = bb_w < bb_w_avg * self.squeeze_ratio
        has_cp = self._recent_changepoint(i)

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
        if side == 1 and trend_val < 0:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and trend_val > 0:
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场 ──
        if side == 0 and in_squeeze and has_cp:
            if trend_val > 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
            elif trend_val < 0:
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
                    and trend_val > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and trend_val < 0):
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
