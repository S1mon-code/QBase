import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.momentum.macd import macd
from indicators.volatility.bollinger import bollinger_bands
from indicators.volume.klinger import klinger
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV187(TimeSeriesStrategy):
    """
    策略简介：MACD交叉+BB%B中轨附近+Klinger成交量确认的短周期多空策略。

    使用指标：
    - MACD(12,26,9): 动量交叉信号，histogram方向确认
    - Bollinger %B(20,2): 价格在布林带中的相对位置，中轨附近入场
    - Klinger(34,55,13): 成交量力度方向确认
    - ATR(14): 止损距离计算

    进场条件（做多）：MACD histogram由负转正 + %B在0.3-0.7 + Klinger > signal
    进场条件（做空）：MACD histogram由正转负 + %B在0.3-0.7 + Klinger < signal

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - MACD histogram反向

    优点：中轨入场避免追高追低，Klinger确认资金方向
    缺点：30min噪音较多，需要严格止损控制
    """
    name = "ag_alltime_v187"
    warmup = 500
    freq = "30min"

    macd_fast: int = 12           # Optuna: 8-16
    macd_slow: int = 26           # Optuna: 20-34
    macd_signal: int = 9          # Optuna: 7-13
    bb_period: int = 20           # Optuna: 14-30
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._macd_line = None
        self._macd_sig = None
        self._macd_hist = None
        self._bb_upper = None
        self._bb_mid = None
        self._bb_lower = None
        self._klinger_line = None
        self._klinger_sig = None
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

        self._macd_line, self._macd_sig, self._macd_hist = macd(
            closes, self.macd_fast, self.macd_slow, self.macd_signal)
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            closes, self.bb_period, 2.0)
        self._klinger_line, self._klinger_sig = klinger(highs, lows, closes, volumes)
        self._atr = atr(highs, lows, closes, period=14)

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
        hist = self._macd_hist[i]
        kl = self._klinger_line[i]
        ks = self._klinger_sig[i]
        bb_u = self._bb_upper[i]
        bb_l = self._bb_lower[i]

        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(hist) or np.isnan(kl) or np.isnan(ks):
            return
        if np.isnan(bb_u) or np.isnan(bb_l) or bb_u == bb_l:
            return

        prev_hist = self._macd_hist[i - 1] if i > 0 else np.nan
        if np.isnan(prev_hist):
            return

        pctb = (price - bb_l) / (bb_u - bb_l)

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

        # 3. Signal exit: MACD histogram flip
        if side == 1 and hist < 0 and prev_hist >= 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and hist > 0 and prev_hist <= 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: MACD crossover near BB middle + Klinger confirmation
        if side == 0:
            near_mid = 0.3 <= pctb <= 0.7
            if prev_hist <= 0 and hist > 0 and near_mid and kl > ks:
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
            elif prev_hist >= 0 and hist < 0 and near_mid and kl < ks:
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
            if self.direction == 1 and hist > 0 and kl > ks:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and hist < 0 and kl < ks:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
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
