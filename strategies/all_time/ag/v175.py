import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.regime.market_state import market_state
from indicators.momentum.rsi import rsi
from indicators.volatility.bollinger import bollinger_bands
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class MarketStateRSIBBStrategy(TimeSeriesStrategy):
    """
    策略简介：Market State regime + RSI + Bollinger Bands的4h多空策略。
    趋势市：RSI回调至BB中轨入场；震荡市：RSI在BB上下轨均值回归。

    使用指标：
    - Market State(20): 市场状态识别（趋势/震荡/突破）
    - RSI(14): 动量振荡，超买超卖判断
    - Bollinger Bands(20,2.0): 波动率通道
    - ATR(14): 止损距离计算

    进场条件（趋势做多）：
    - Market State = 趋势上涨（state > 0）
    - RSI回调至30-50区间（回调入场）
    - 价格在BB中轨附近

    进场条件（震荡做多）：
    - Market State = 震荡（state = 0）
    - RSI < 30（超卖）
    - 价格触及BB下轨

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - RSI超买/超卖反转

    优点：根据市场状态自适应切换策略逻辑，趋势追动量/震荡做反转
    缺点：市场状态判断滞后，状态切换频繁时两头挨打
    """
    name = "v175_market_state_rsi_bb"
    warmup = 500
    freq = "4h"

    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    ms_period: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._ms = None
        self._rsi = None
        self._bb_upper = None
        self._bb_mid = None
        self._bb_lower = None
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

        self._atr = atr(highs, lows, closes, period=14)
        self._rsi = rsi(closes, period=self.rsi_period)
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            closes, period=self.bb_period, num_std=self.bb_std)
        self._ms, _ = market_state(closes, volumes, oi, period=self.ms_period)

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

        ms_val = self._ms[i]
        rsi_val = self._rsi[i]
        bb_u = self._bb_upper[i]
        bb_m = self._bb_mid[i]
        bb_l = self._bb_lower[i]
        atr_val = self._atr[i]
        if np.isnan(ms_val) or np.isnan(rsi_val) or np.isnan(bb_u) or np.isnan(atr_val):
            return

        self.bars_since_last_scale += 1
        is_trending = ms_val != 0  # non-zero = trending
        trend_up = ms_val > 0
        trend_down = ms_val < 0

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

        # ── 3. RSI反转退出 ──
        if side == 1 and rsi_val > 75:
            context.close_long()
            self._reset_state()
        elif side == -1 and rsi_val < 25:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if is_trending and trend_up:
                # Trending long: RSI pullback near BB mid
                if 30 < rsi_val < 50 and price <= bb_m * 1.01:
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
            elif is_trending and trend_down:
                # Trending short: RSI pullback near BB mid
                if 50 < rsi_val < 70 and price >= bb_m * 0.99:
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
            elif not is_trending:
                # Ranging: mean reversion at BB bands
                if rsi_val < 30 and price <= bb_l:
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
                elif rsi_val > 70 and price >= bb_u:
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
                    and rsi_val < 60):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and rsi_val > 40):
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
