import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.volatility.hurst import hurst_exponent
from indicators.trend.ema import ema
from indicators.volatility.bollinger import bollinger_bands

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV12(TimeSeriesStrategy):
    """
    策略简介：基于Hurst指数区分趋势/均值回归，自适应交易模式。

    使用指标：
    - Hurst Exponent(100): H>0.6趋势跟踪，H<0.4均值回归
    - EMA(30): 趋势方向判断
    - Bollinger Bands(20,2): 均值回归的超买超卖参考
    - ATR(14): 止损距离计算

    进场条件（做多-趋势）：H>0.6, 价格>EMA30
    进场条件（做空-趋势）：H>0.6, 价格<EMA30
    进场条件（做多-回归）：H<0.4, 价格<BB下轨
    进场条件（做空-回归）：H<0.4, 价格>BB上轨

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Hurst指数跨越中性区间

    优点：统计理论支撑，自适应趋势/震荡切换
    缺点：Hurst估计在小样本下不稳定，需要较长lookback
    """
    name = "ag_alltime_v12"
    warmup = 400
    freq = "daily"

    hurst_lag: int = 100           # Optuna: 50-200
    trend_hurst: float = 0.6     # Optuna: 0.55-0.70
    revert_hurst: float = 0.4   # Optuna: 0.30-0.45
    ema_period: int = 30          # Optuna: 15-50
    bb_period: int = 20           # Optuna: 15-30
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._hurst = None
        self._ema = None
        self._bb_upper = None
        self._bb_lower = None
        self._bb_mid = None
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
        self.mode = 0  # 1=trend, 2=mean_revert

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._atr = atr(highs, lows, closes, period=14)
        self._hurst = hurst_exponent(closes, max_lag=self.hurst_lag)
        self._ema = ema(closes, period=self.ema_period)
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            closes, period=self.bb_period
        )

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
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        h = self._hurst[i]
        ema_val = self._ema[i]
        bb_up = self._bb_upper[i]
        bb_lo = self._bb_lower[i]
        bb_mid = self._bb_mid[i]
        if np.isnan(h) or np.isnan(ema_val) or np.isnan(bb_up):
            return

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

        # 3. Signal-based exit
        if side != 0:
            # Mode mismatch
            if self.mode == 1 and h < self.revert_hurst:
                if side == 1:
                    context.close_long()
                else:
                    context.close_short()
                self._reset_state()
                return
            if self.mode == 2 and h > self.trend_hurst:
                if side == 1:
                    context.close_long()
                else:
                    context.close_short()
                self._reset_state()
                return
            # Mean-revert target: return to BB mid
            if self.mode == 2:
                if side == 1 and price >= bb_mid:
                    context.close_long()
                    self._reset_state()
                    return
                if side == -1 and price <= bb_mid:
                    context.close_short()
                    self._reset_state()
                    return

        # 4. Entry
        if side == 0:
            if h > self.trend_hurst:
                if price > ema_val:
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
                        self.mode = 1
                elif price < ema_val:
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
                        self.mode = 1
            elif h < self.revert_hurst:
                if price < bb_lo:
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
                        self.mode = 2
                elif price > bb_up:
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
                        self.mode = 2

        # 5. Scale-in (trend mode only)
        elif side != 0 and self.mode == 1 and self._should_add(price, atr_val):
            if self.direction == 1 and h > self.trend_hurst:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and h > self.trend_hurst:
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
        self.mode = 0
