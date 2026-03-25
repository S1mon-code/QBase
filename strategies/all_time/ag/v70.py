import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.regime.volatility_clustering import vol_clustering
from indicators.momentum.stochastic import stochastic
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class HerdBehaviorStochastic(TimeSeriesStrategy):
    """
    策略简介：波动率聚集（羊群行为代理）+ 随机指标。聚集时跟随，分散时反转。

    使用指标：
    - vol_clustering(60): 波动率聚集度 (高=羊群跟风, 低=分散独立)
    - stochastic(14, 3): 随机指标 %K, %D
    - ATR(14): 止损距离计算

    进场条件（羊群模式 - clustering > 0.3）：
    - %K 从超卖区域上穿 %D → 做多（跟随众人）
    - %K 从超买区域下穿 %D → 做空

    进场条件（分散模式 - clustering < 0.1）：
    - %K > 80 超买 → 做空（逆向操作）
    - %K < 20 超卖 → 做多

    出场条件：
    - ATR 追踪止损 / 分层止盈
    - 模式切换时平仓

    优点：区分趋势和反转市场环境
    缺点：1h 频率噪音多，聚集度判断可能滞后
    """
    name = "v70_herd_behavior_stochastic"
    warmup = 500
    freq = "1h"

    vc_period: int = 60
    herd_threshold: float = 0.3
    dispersed_threshold: float = 0.1
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._clustering = None
        self._stoch_k = None
        self._stoch_d = None
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
        self.entry_mode = 0  # 1=herd, -1=dispersed

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._clustering, _ = vol_clustering(closes, period=self.vc_period)
        self._stoch_k, self._stoch_d = stochastic(highs, lows, closes,
                                                     k_period=self.stoch_k,
                                                     d_period=self.stoch_d)
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

        clust = self._clustering[i]
        sk = self._stoch_k[i]
        sd = self._stoch_d[i]
        atr_val = self._atr[i]
        if np.isnan(clust) or np.isnan(sk) or np.isnan(sd) or np.isnan(atr_val):
            return

        is_herd = clust > self.herd_threshold
        is_dispersed = clust < self.dispersed_threshold

        # Previous bar stochastic for crossover detection
        prev_sk = self._stoch_k[i - 1] if i > 0 else np.nan
        prev_sd = self._stoch_d[i - 1] if i > 0 else np.nan
        if np.isnan(prev_sk) or np.isnan(prev_sd):
            prev_sk = sk
            prev_sd = sd

        k_cross_up = prev_sk <= prev_sd and sk > sd
        k_cross_down = prev_sk >= prev_sd and sk < sd

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
        if side == 1:
            if self.entry_mode == 1 and sk > self.stoch_overbought:
                context.close_long()
                self._reset_state()
                return
            if self.entry_mode == -1 and sk > 50:
                context.close_long()
                self._reset_state()
                return
        elif side == -1:
            if self.entry_mode == 1 and sk < self.stoch_oversold:
                context.close_short()
                self._reset_state()
                return
            if self.entry_mode == -1 and sk < 50:
                context.close_short()
                self._reset_state()
                return

        # ── 4. 入场逻辑 ──
        if side == 0:
            if is_herd:
                if k_cross_up and sk < 50:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.buy(base_lots)
                        self._set_entry(price, price - self.atr_stop_mult * atr_val, 1)
                elif k_cross_down and sk > 50:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.sell(base_lots)
                        self._set_entry(price, price + self.atr_stop_mult * atr_val, 1)
            elif is_dispersed:
                if sk < self.stoch_oversold:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.buy(base_lots)
                        self._set_entry(price, price - self.atr_stop_mult * atr_val, -1)
                elif sk > self.stoch_overbought:
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
