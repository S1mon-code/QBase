import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.trend.adx import adx_with_di
from indicators.volatility.historical_vol import historical_volatility

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _market_state(adx_arr, hvol_arr, closes, period=20):
    """Classify market state: 1=trending_up, 2=trending_down, 0=ranging, 3=volatile.

    Uses ADX for trend strength, HV for volatility regime, price direction for bias.
    """
    n = len(closes)
    state = np.full(n, 0, dtype=np.int32)
    strength = np.full(n, np.nan, dtype=np.float64)

    for i in range(period, n):
        a = adx_arr[i]
        hv = hvol_arr[i]
        if np.isnan(a) or np.isnan(hv):
            continue

        # Price direction over period
        if closes[i] > closes[i - period]:
            price_dir = 1
        elif closes[i] < closes[i - period]:
            price_dir = -1
        else:
            price_dir = 0

        # HV percentile approximation (using rolling mean)
        hv_window = hvol_arr[max(0, i - 120):i + 1]
        hv_valid = hv_window[~np.isnan(hv_window)]
        if len(hv_valid) < 10:
            continue
        hv_pct = np.sum(hv_valid < hv) / len(hv_valid)

        if a > 25:
            if price_dir > 0:
                state[i] = 1  # trending up
            else:
                state[i] = 2  # trending down
            strength[i] = min(1.0, a / 50.0)
        elif hv_pct > 0.8:
            state[i] = 3  # volatile/choppy
            strength[i] = hv_pct
        else:
            state[i] = 0  # ranging
            strength[i] = 1.0 - a / 25.0

    return state, strength


class StrategyV4(TimeSeriesStrategy):
    """
    策略简介：基于多维市场状态分类的方向性交易策略。

    使用指标：
    - ADX(14) + DI: 趋势强度和方向
    - Historical Volatility(20): 波动率状态判断
    - Market State Classification: 综合状态分类
    - ATR(14): 止损距离计算

    进场条件（做多）：市场状态=trending_up(1)，强度>0.4，+DI>-DI
    进场条件（做空）：市场状态=trending_down(2)，强度>0.4，-DI>+DI

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 状态切换为反向或震荡

    优点：综合多维信息判断市场状态，避免在震荡市中交易
    缺点：状态分类可能在边界处不稳定
    """
    name = "ag_alltime_v4"
    warmup = 400
    freq = "daily"

    adx_period: int = 14          # Optuna: 10-20
    hvol_period: int = 20         # Optuna: 10-40
    strength_thresh: float = 0.4  # Optuna: 0.2-0.6
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._state = None
        self._strength = None
        self._plus_di = None
        self._minus_di = None
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

        self._atr = atr(highs, lows, closes, period=14)
        adx_arr, self._plus_di, self._minus_di = adx_with_di(
            highs, lows, closes, period=self.adx_period
        )
        hvol_arr = historical_volatility(closes, period=self.hvol_period)
        self._state, self._strength = _market_state(adx_arr, hvol_arr, closes)

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
        state_val = self._state[i]
        str_val = self._strength[i]
        if np.isnan(str_val):
            return
        pdi = self._plus_di[i]
        mdi = self._minus_di[i]
        if np.isnan(pdi) or np.isnan(mdi):
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
        if side == 1 and (state_val == 2 or state_val == 0):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (state_val == 1 or state_val == 0):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if state_val == 1 and str_val > self.strength_thresh and pdi > mdi:
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
            elif state_val == 2 and str_val > self.strength_thresh and mdi > pdi:
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
            if self.direction == 1 and state_val == 1:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and state_val == 2:
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
