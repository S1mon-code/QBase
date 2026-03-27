import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.hma import hma
from indicators.momentum.tsi import tsi
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV88(TimeSeriesStrategy):
    """
    策略简介：HMA趋势方向 + TSI动量确认的4h多头策略。

    使用指标：
    - HMA(20): Hull均线方向判断，上升为多头
    - TSI(25,13,7): True Strength Index动量确认
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - HMA在上升（当前HMA > 2bar前HMA）
    - TSI > 0 且 TSI > 信号线

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - TSI < 信号线（动量减弱）

    优点：HMA响应快、平滑好，TSI双重平滑抗噪
    缺点：HMA参数敏感，短周期HMA易被急跌打出
    """
    name = "medium_trend_v88"
    warmup = 200
    freq = "4h"

    hma_period: int = 20       # Optuna: 12-30
    tsi_long: int = 25
    tsi_short: int = 13
    tsi_signal: int = 7
    atr_stop_mult: float = 3.0  # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._hma = None
        self._tsi = None
        self._tsi_sig = None
        self._atr = None
        self._avg_volume = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._hma = hma(closes, self.hma_period)
        self._tsi, self._tsi_sig = tsi(closes, long=self.tsi_long, short=self.tsi_short, signal=self.tsi_signal)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        hma_val = self._hma[i]
        tsi_val = self._tsi[i]
        tsi_sig = self._tsi_sig[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(hma_val) or np.isnan(tsi_val) or np.isnan(tsi_sig):
            return
        hma_prev = self._hma[i - 2] if i >= 2 else np.nan
        if np.isnan(hma_prev):
            return

        hma_rising = hma_val > hma_prev

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

        # 2. Tiered profit-taking
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

        # 3. Signal exit: TSI below signal
        if side == 1 and tsi_val < tsi_sig:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and hma_rising and tsi_val > 0 and tsi_val > tsi_sig:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, hma_rising, tsi_val, tsi_sig):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, hma_rising, tsi_val, tsi_sig):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if not hma_rising or tsi_val <= tsi_sig:
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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
