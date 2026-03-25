import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.regime.market_state import market_state
from indicators.momentum.rsi import rsi
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class MarketStateRSIStrategy(TimeSeriesStrategy):
    """
    策略简介：Market State状态识别 + RSI分场景交易的多空策略。

    使用指标：
    - Market State(20): 多维市场状态分类器（0=quiet, 1=trending_up,
      2=trending_down, 3=volatile_range, 4=breakout）
    - RSI(14): 动量/超买超卖
    - ATR(14): 止损距离计算

    策略逻辑（分状态交易）：
    - 趋势状态(1/2): RSI回调入场（趋势中找pullback）
      - trending_up + RSI < 40: 做多（回调买入）
      - trending_down + RSI > 60: 做空（反弹做空）
    - 震荡状态(0/3): RSI极值反转交易
      - quiet/volatile_range + RSI < 25: 做多（超卖反转）
      - quiet/volatile_range + RSI > 75: 做空（超买反转）
    - 突破状态(4): RSI方向跟随
      - breakout + RSI > 50: 做多
      - breakout + RSI < 50: 做空

    出场条件：
    - ATR追踪止损 / 分层止盈 / 状态变化 + RSI反转

    优点：分状态交易自适应不同市场环境，RSI规则清晰简单
    缺点：状态判断可能滞后，状态切换频繁时产生噪音
    """
    name = "v60_market_state_rsi"
    warmup = 500
    freq = "4h"

    state_period: int = 20
    rsi_period: int = 14
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._state = None
        self._state_conf = None
        self._rsi = None
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
        self.entry_state = -1  # Track which state we entered in

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()
        n = len(closes)

        self._state, self._state_conf = market_state(
            closes, volumes, oi, period=self.state_period)
        self._rsi = rsi(closes, self.rsi_period)
        self._atr = atr(highs, lows, closes, period=self.atr_period)

        window = 20
        self._avg_volume = np.full_like(volumes, np.nan)
        for idx in range(window, n):
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

        state_val = self._state[i]
        rsi_val = self._rsi[i]
        atr_val = self._atr[i]
        if np.isnan(state_val) or np.isnan(rsi_val) or np.isnan(atr_val) or atr_val <= 0:
            return

        state = int(state_val)
        self.bars_since_last_scale += 1

        is_trending_up = state == 1
        is_trending_down = state == 2
        is_ranging = state in (0, 3)
        is_breakout = state == 4

        # Determine entry signals based on state
        go_long = False
        go_short = False
        if is_trending_up and rsi_val < 40:
            go_long = True  # Pullback in uptrend
        elif is_trending_down and rsi_val > 60:
            go_short = True  # Rally in downtrend
        elif is_ranging and rsi_val < 25:
            go_long = True  # Oversold reversal
        elif is_ranging and rsi_val > 75:
            go_short = True  # Overbought reversal
        elif is_breakout and rsi_val > 50:
            go_long = True  # Breakout bullish
        elif is_breakout and rsi_val < 50:
            go_short = True  # Breakout bearish

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

        # ── 3. State-based exit ──
        # Trending entries: exit on RSI reversal
        if side == 1 and self.entry_state in (1, 4) and rsi_val > 80:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and self.entry_state in (2, 4) and rsi_val < 20:
            context.close_short()
            self._reset_state()
            return
        # Ranging entries: exit on RSI mean reversion completion
        if side == 1 and self.entry_state in (0, 3) and rsi_val > 60:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and self.entry_state in (0, 3) and rsi_val < 40:
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场 ──
        if side == 0:
            if go_long:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
                    self.entry_state = state
            elif go_short:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
                    self.entry_state = state

        # ── 5. 加仓 (only in trending / breakout states) ──
        elif side == 1 and self.position_scale < MAX_SCALE and (is_trending_up or is_breakout):
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and rsi_val < 60):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE and (is_trending_down or is_breakout):
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
        self.entry_state = -1
