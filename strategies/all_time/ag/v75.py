import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.spread.correlation_regime import correlation_regime
from indicators.spread.relative_strength import relative_strength
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class CrossMarketRelativeStrength(TimeSeriesStrategy):
    """
    策略简介：跨市场相关性状态 + 白银相对强度。在风险偏好上升时做多强势白银。

    使用指标：
    - correlation_regime(20, 60): 金银相关性状态变化（快慢相关性差 = 状态切换）
    - relative_strength(20): AG 相对 AU 的强度 + 新高
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - 相关性 divergence > 0（关系变化，通常意味着白银独立走强）
    - AG 相对强度动量 > 0 且创新高

    进场条件（做空）：
    - 相关性 divergence < 0
    - AG 相对强度动量 < 0

    出场条件：
    - ATR 追踪止损 / 分层止盈
    - 相对强度反转

    优点：利用跨市场信息，捕捉白银独立行情
    缺点：需要 AU 数据，相关性可能长期稳定导致信号稀少
    """
    name = "v75_cross_market_relative_strength"
    warmup = 250
    freq = "daily"

    corr_fast: int = 20
    corr_slow: int = 60
    rs_period: int = 20
    divergence_threshold: float = 0.1
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._corr_div = None
        self._rs_mom = None
        self._rs_new_high = None
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

        try:
            au_closes = context.load_auxiliary_close("AU")
        except Exception:
            au_closes = np.full_like(closes, np.nan)

        _, _, self._corr_div = correlation_regime(
            closes, au_closes, fast=self.corr_fast, slow=self.corr_slow)
        _, self._rs_mom, self._rs_new_high = relative_strength(
            closes, au_closes, period=self.rs_period)
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

        corr_div = self._corr_div[i]
        rs_mom = self._rs_mom[i]
        atr_val = self._atr[i]
        if np.isnan(corr_div) or np.isnan(rs_mom) or np.isnan(atr_val):
            return

        rs_nh = self._rs_new_high[i] if i < len(self._rs_new_high) else False
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
        if side == 1 and rs_mom < 0:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and rs_mom > 0:
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场逻辑 ──
        if side == 0:
            if corr_div > self.divergence_threshold and rs_mom > 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self._set_entry(price, price - self.atr_stop_mult * atr_val)
            elif corr_div < -self.divergence_threshold and rs_mom < 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self._set_entry(price, price + self.atr_stop_mult * atr_val)

        # ── 5. 加仓逻辑 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and rs_mom > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and rs_mom < 0):
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
