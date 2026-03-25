import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.momentum.stochastic import stochastic
from indicators.spread.gold_silver_ratio import gold_silver_ratio
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StochasticGoldSilverStrategy(TimeSeriesStrategy):
    """
    策略简介：Stochastic 超买超卖 + 金银比价差过滤的多空策略。

    使用指标：
    - Stochastic(14,3): 超买超卖区间判断
    - Gold/Silver Ratio(60): 金银比z-score，高z=银相对便宜，利于做多银
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Stochastic %K < 20 且上穿 %D（超卖金叉）
    - 金银比 z-score > 0（银相对金偏低，有上涨空间）

    进场条件（做空）：
    - Stochastic %K > 80 且下穿 %D（超买死叉）
    - 金银比 z-score < 0（银相对金偏高，有下跌空间）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Stochastic 反向交叉

    优点：跨品种信息增强白银方向判断
    缺点：依赖AU数据同步，金银比周期可能与交易频率不匹配
    """
    name = "v89_stoch_gold_silver"
    warmup = 300
    freq = "4h"

    k_period: int = 14
    d_period: int = 3
    gs_period: int = 60
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._stoch_k = None
        self._stoch_d = None
        self._gs_ratio = None
        self._gs_zscore = None
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

        self._stoch_k, self._stoch_d = stochastic(
            highs, lows, closes, self.k_period, self.d_period)
        self._atr = atr(highs, lows, closes, period=14)

        # Load AU auxiliary close for gold/silver ratio
        au_closes = context.load_auxiliary_close("AU")
        if au_closes is not None and len(au_closes) == len(closes):
            self._gs_ratio, self._gs_zscore = gold_silver_ratio(
                au_closes, closes, self.gs_period)
        else:
            # Fallback: no AU data available, use neutral values
            self._gs_ratio = np.full(len(closes), np.nan)
            self._gs_zscore = np.zeros(len(closes))

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

        k_val = self._stoch_k[i]
        d_val = self._stoch_d[i]
        gs_z = self._gs_zscore[i]
        atr_val = self._atr[i]
        if np.isnan(k_val) or np.isnan(d_val) or np.isnan(atr_val):
            return
        if np.isnan(gs_z):
            gs_z = 0.0

        self.bars_since_last_scale += 1

        prev_k = self._stoch_k[i - 1] if i > 0 else np.nan
        prev_d = self._stoch_d[i - 1] if i > 0 else np.nan
        if np.isnan(prev_k) or np.isnan(prev_d):
            return

        k_cross_up = prev_k <= prev_d and k_val > d_val
        k_cross_down = prev_k >= prev_d and k_val < d_val

        long_signal = k_cross_up and k_val < self.stoch_oversold and gs_z > 0
        short_signal = k_cross_down and k_val > self.stoch_overbought and gs_z < 0

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
        if side == 1 and k_cross_down and k_val > self.stoch_overbought:
            context.close_long()
            self._reset_state()
        elif side == -1 and k_cross_up and k_val < self.stoch_oversold:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if long_signal:
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
            elif short_signal:
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
                    and k_val < 80 and gs_z > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and k_val > 20 and gs_z < 0):
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
