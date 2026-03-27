import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.hmm_regime import hmm_regime
from indicators.regime.market_state import market_state
from indicators.momentum.rsi import rsi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV115(TimeSeriesStrategy):
    """
    策略简介：HMM隐状态 + Market State双重趋势确认下的RSI回调入场策略。

    使用指标：
    - HMM Regime(252, 3): 隐马尔可夫模型识别市场隐状态
    - Market State(20): 多维市场状态分类器（趋势/震荡/突破）
    - RSI(14): 回调入场信号
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - HMM状态=1(上涨趋势) + Market State=1(trending_up)
    - RSI回调至30-50区间（趋势中的回调买入）
    进场条件（做空）：
    - HMM状态=2(下跌趋势) + Market State=2(trending_down)
    - RSI反弹至50-70区间（趋势中的反弹卖出）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - HMM或Market State不再确认趋势

    优点：三重确认（HMM+状态+RSI回调）大幅提高信号可靠性
    缺点：多重过滤可能导致入场机会过少
    """
    name = "ag_alltime_v115"
    warmup = 800
    freq = "4h"

    hmm_period: int = 252         # Optuna: 120-504
    ms_period: int = 20           # Optuna: 10-30
    rsi_period: int = 14          # Optuna: 10-20
    rsi_pullback_lo: float = 35.0 # Optuna: 25-45
    rsi_pullback_hi: float = 65.0 # Optuna: 55-75
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._hmm_labels = None
        self._hmm_probs = None
        self._ms_state = None
        self._ms_conf = None
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
        self.direction = 0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()

        self._atr = atr(highs, lows, closes, period=14)
        self._rsi = rsi(closes, period=self.rsi_period)

        # HMM regime: states 0=sideways, 1=up, 2=down
        self._hmm_labels, self._hmm_probs, _ = hmm_regime(
            closes, n_states=3, period=self.hmm_period
        )

        # Market state: 0=quiet, 1=trending_up, 2=trending_down, 3=volatile, 4=breakout
        self._ms_state, self._ms_conf = market_state(
            closes, volumes, oi, period=self.ms_period
        )

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        rsi_val = self._rsi[i]
        hmm_lbl = self._hmm_labels[i]
        ms_val = self._ms_state[i]
        if np.isnan(rsi_val) or np.isnan(hmm_lbl) or np.isnan(ms_val):
            return

        hmm_state = int(hmm_lbl)
        ms_state = int(ms_val)

        self.bars_since_last_scale += 1

        # 1. Stop loss
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

        # 2. Tiered profit-taking
        if side != 0 and self.entry_price > 0:
            if side == 1:
                profit_atr = (price - self.entry_price) / atr_val
            else:
                profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_3atr = True
                return

        # Determine trending conditions
        bullish_trend = hmm_state == 1 and ms_state == 1
        bearish_trend = hmm_state == 2 and ms_state == 2

        # 3. Signal-based exit: trend no longer confirmed
        if side == 1 and not bullish_trend:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and not bearish_trend:
            context.close_short()
            self._reset_state()
            return

        # Re-read position
        side, lots = context.position

        # 4. Entry: confirmed trend + RSI pullback
        if side == 0:
            if bullish_trend and rsi_val < self.rsi_pullback_lo:
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
            elif bearish_trend and rsi_val > self.rsi_pullback_hi:
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

        # 5. Scale-in
        elif side != 0 and self._should_add(price, atr_val):
            signal_ok = (self.direction == 1 and bullish_trend and rsi_val < 50) or \
                        (self.direction == -1 and bearish_trend and rsi_val > 50)
            if signal_ok:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    if self.direction == 1:
                        context.buy(add_lots)
                    else:
                        context.sell(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1 and price < self.entry_price + atr_val:
            return False
        if self.direction == -1 and price > self.entry_price - atr_val:
            return False
        return True

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

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
