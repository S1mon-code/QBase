import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.ema import ema_cross
from indicators.trend.adx import adx
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV46(TimeSeriesStrategy):
    """
    策略简介：EMA交叉 + ADX趋势强度过滤的30min经典趋势策略。

    使用指标：
    - EMA Cross(10, 30): 快慢均线交叉产生趋势信号
    - ADX(14): 趋势强度过滤，>20时确认趋势存在
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - EMA交叉信号 = 1（快线上穿慢线）
    - ADX > adx_threshold

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - EMA交叉信号 = -1（快线下穿慢线）

    优点：经典双均线系统，信号明确，ADX过滤震荡
    缺点：均线系统天然滞后，快速反转时亏损较大
    """
    name = "medium_trend_v46"
    warmup = 600
    freq = "30min"

    ema_fast: int = 10               # Optuna: 5-20
    ema_slow: int = 30               # Optuna: 20-50
    adx_period: int = 14
    adx_threshold: float = 20.0      # Optuna: 15-35
    atr_stop_mult: float = 3.0       # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._ema_signal = None
        self._adx = None
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

        self._ema_signal = ema_cross(closes, self.ema_fast, self.ema_slow)
        self._adx = adx(highs, lows, closes, period=self.adx_period)
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
        ema_sig = self._ema_signal[i]
        adx_val = self._adx[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(ema_sig) or np.isnan(adx_val):
            return

        self.bars_since_last_scale += 1

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

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

        if side == 1 and ema_sig == -1:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and ema_sig == 1 and adx_val > self.adx_threshold:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, ema_sig, adx_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, ema_sig, adx_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if ema_sig != 1 or adx_val < self.adx_threshold:
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
