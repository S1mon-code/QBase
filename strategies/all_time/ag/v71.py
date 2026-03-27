import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.regime.runs_test import runs_test
from indicators.volatility.hurst import hurst_exponent
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class NoiseRegimeHurst(TimeSeriesStrategy):
    """
    策略简介：游程检验 + Hurst 指数双重确认趋势/均值回归环境。

    使用指标：
    - runs_test(60): 随机性检验 (z<-2=趋势, z>2=均值回归)
    - hurst_exponent(20): Hurst 指数 (H>0.5=趋势, H<0.5=均值回归)
    - ATR(14): 止损距离计算

    进场条件（趋势跟随）：
    - runs_test is_trending = True 且 Hurst > 0.6
    - 价格在近期方向上运动（使用简单动量判断方向）

    进场条件（均值回归）：
    - runs_test is_mean_reverting = True 且 Hurst < 0.4
    - 价格偏离均值较远时反向入场

    出场条件：
    - ATR 追踪止损 / 分层止盈
    - 双指标不再一致确认

    优点：两种独立方法交叉验证，信号更可靠
    缺点：双重确认可能过于保守，错过部分机会
    """
    name = "v71_noise_regime_hurst"
    warmup = 250
    freq = "daily"

    runs_period: int = 60
    hurst_max_lag: int = 20
    hurst_trend: float = 0.6
    hurst_mr: float = 0.4
    mom_period: int = 20
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._runs_z = None
        self._is_trending = None
        self._is_mr = None
        self._hurst = None
        self._atr = None
        self._closes = None
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
        self.entry_mode = 0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._closes = closes
        self._runs_z, self._is_trending, self._is_mr = runs_test(
            closes, period=self.runs_period)
        self._hurst = hurst_exponent(closes, max_lag=self.hurst_max_lag)
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

        hurst_val = self._hurst[i]
        is_trend = self._is_trending[i]
        is_mr = self._is_mr[i]
        atr_val = self._atr[i]
        if np.isnan(hurst_val) or np.isnan(atr_val):
            return

        # Simple momentum for direction
        mom_dir = 0
        if i >= self.mom_period:
            mom = self._closes[i] - self._closes[i - self.mom_period]
            mom_dir = 1 if mom > 0 else (-1 if mom < 0 else 0)

        # SMA for mean reversion reference
        sma = np.nan
        if i >= self.mom_period:
            sma = np.mean(self._closes[i - self.mom_period:i + 1])

        double_trend = is_trend and hurst_val > self.hurst_trend
        double_mr = is_mr and hurst_val < self.hurst_mr

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
        if side != 0:
            if self.entry_mode == 1 and not double_trend:
                if side == 1:
                    context.close_long()
                else:
                    context.close_short()
                self._reset_state()
                return
            if self.entry_mode == -1 and not double_mr:
                if side == 1:
                    context.close_long()
                else:
                    context.close_short()
                self._reset_state()
                return

        # ── 4. 入场逻辑 ──
        if side == 0:
            if double_trend and mom_dir != 0:
                if mom_dir == 1:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.buy(base_lots)
                        self._set_entry(price, price - self.atr_stop_mult * atr_val, 1)
                else:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.sell(base_lots)
                        self._set_entry(price, price + self.atr_stop_mult * atr_val, 1)
            elif double_mr and not np.isnan(sma):
                deviation = (price - sma) / atr_val
                if deviation < -1.5:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.buy(base_lots)
                        self._set_entry(price, price - self.atr_stop_mult * atr_val, -1)
                elif deviation > 1.5:
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
