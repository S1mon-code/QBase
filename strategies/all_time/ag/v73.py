import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.regime.trend_persistence import trend_persistence
from indicators.spread.gold_silver_ratio import gold_silver_ratio
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class TrendMaturityGoldSilver(TimeSeriesStrategy):
    """
    策略简介：趋势成熟度 + 金银比策略。早期趋势中交易比值背离，晚期交易比值回归。

    使用指标：
    - trend_persistence(20, 60): 趋势持续性（代理趋势成熟度）
    - gold_silver_ratio(60): 金银价格比 + z-score
    - ATR(14): 止损距离计算

    进场条件（早期趋势 - persistence 低但上升中）：
    - 金银比 z-score > 1.5（白银相对便宜）→ 做多 AG
    - 金银比 z-score < -1.5（白银相对贵）→ 做空 AG

    进场条件（成熟趋势 - persistence 高且稳定）：
    - 金银比 z-score < -1.0（白银相对贵 → 回归 → 做多 AG）
    - 金银比 z-score > 1.0（白银相对便宜 → 回归 → 做空 AG）

    出场条件：
    - ATR 追踪止损 / 分层止盈
    - z-score 回到 ±0.5 内

    优点：利用金银比的均值回归特性，结合趋势成熟度自适应
    缺点：需要 AU 数据，跨品种关系可能失效
    """
    name = "v73_trend_maturity_gold_silver"
    warmup = 250
    freq = "daily"

    tp_max_lag: int = 20
    tp_period: int = 60
    early_persistence: float = 0.3
    mature_persistence: float = 0.6
    gsr_period: int = 60
    zscore_diverge: float = 1.5
    zscore_revert: float = 1.0
    zscore_exit: float = 0.5
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._persistence = None
        self._gsr_ratio = None
        self._gsr_zscore = None
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

        # Load gold (AU) auxiliary close
        try:
            au_closes = context.load_auxiliary_close("AU")
        except Exception:
            au_closes = np.full_like(closes, np.nan)

        self._persistence, _ = trend_persistence(closes, max_lag=self.tp_max_lag,
                                                   period=self.tp_period)
        self._gsr_ratio, self._gsr_zscore = gold_silver_ratio(
            au_closes, closes, period=self.gsr_period)
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

        pers = self._persistence[i]
        zscore = self._gsr_zscore[i]
        atr_val = self._atr[i]
        if np.isnan(pers) or np.isnan(zscore) or np.isnan(atr_val):
            return

        is_early = pers < self.early_persistence
        is_mature = pers > self.mature_persistence
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
        if side == 1 and abs(zscore) < self.zscore_exit:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and abs(zscore) < self.zscore_exit:
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场逻辑 ──
        if side == 0:
            if is_early:
                # Early trend: trade divergence
                if zscore > self.zscore_diverge:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.buy(base_lots)
                        self._set_entry(price, price - self.atr_stop_mult * atr_val)
                elif zscore < -self.zscore_diverge:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.sell(base_lots)
                        self._set_entry(price, price + self.atr_stop_mult * atr_val)
            elif is_mature:
                # Mature trend: trade reversion
                if zscore < -self.zscore_revert:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.buy(base_lots)
                        self._set_entry(price, price - self.atr_stop_mult * atr_val)
                elif zscore > self.zscore_revert:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.sell(base_lots)
                        self._set_entry(price, price + self.atr_stop_mult * atr_val)

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

    def _set_entry(self, price, stop):
        self.entry_price = price
        self.stop_price = stop
        self.highest_since_entry = price
        self.lowest_since_entry = price
        self.position_scale = 1
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
