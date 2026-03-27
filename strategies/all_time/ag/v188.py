import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr
from indicators.volume.oi_divergence import oi_divergence

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV188(TimeSeriesStrategy):
    """
    策略简介：Supertrend趋势方向 + ATR通道波动过滤 + OI背离确认的趋势跟踪策略。

    使用指标：
    - Supertrend(10, 3.0): 趋势方向判断
    - ATR(14) + SMA: 构建ATR通道，价格在通道内视为正常趋势
    - OI Divergence(20): 持仓量背离检测，确认趋势健康度
    - ATR(14): 止损距离计算

    进场条件（做多）：Supertrend=1 + 价格在ATR通道上半部 + OI背离非负
    进场条件（做空）：Supertrend=-1 + 价格在ATR通道下半部 + OI背离非正

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - Supertrend方向反转

    优点：Supertrend提供清晰信号，ATR通道过滤极端位入场，OI确认资金支持
    缺点：震荡市Supertrend频繁翻转，OI数据可能滞后
    """
    name = "ag_alltime_v188"
    warmup = 200
    freq = "4h"

    st_period: int = 10           # Optuna: 7-15
    st_multiplier: float = 3.0   # Optuna: 2.0-5.0
    atr_chan_period: int = 20     # Optuna: 14-30
    atr_chan_mult: float = 1.5    # Optuna: 1.0-2.5
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._st_line = None
        self._st_dir = None
        self._atr = None
        self._oi_div = None
        self._chan_mid = None
        self._chan_upper = None
        self._chan_lower = None
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

        self._st_line, self._st_dir = supertrend(
            highs, lows, closes, self.st_period, self.st_multiplier)
        self._atr = atr(highs, lows, closes, period=14)
        self._oi_div = oi_divergence(closes, oi, period=20)

        # Build ATR channel: SMA(close, period) +/- mult * ATR
        n = len(closes)
        self._chan_mid = np.full(n, np.nan)
        self._chan_upper = np.full(n, np.nan)
        self._chan_lower = np.full(n, np.nan)
        p = self.atr_chan_period
        for idx in range(p - 1, n):
            self._chan_mid[idx] = np.mean(closes[idx - p + 1:idx + 1])
        for idx in range(n):
            if not np.isnan(self._chan_mid[idx]) and not np.isnan(self._atr[idx]):
                self._chan_upper[idx] = self._chan_mid[idx] + self.atr_chan_mult * self._atr[idx]
                self._chan_lower[idx] = self._chan_mid[idx] - self.atr_chan_mult * self._atr[idx]

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
        st_dir = self._st_dir[i]
        oi_d = self._oi_div[i]
        chan_u = self._chan_upper[i]
        chan_l = self._chan_lower[i]
        chan_m = self._chan_mid[i]

        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(st_dir):
            return
        if np.isnan(oi_d) or np.isnan(chan_u) or np.isnan(chan_l):
            return

        prev_dir = self._st_dir[i - 1] if i > 0 else np.nan
        if np.isnan(prev_dir):
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

        # 3. Signal exit: Supertrend flip
        if side == 1 and st_dir == -1:
            context.close_long()
            self._reset_state()
        elif side == -1 and st_dir == 1:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry
        if side == 0:
            price_above_mid = price > chan_m
            price_below_mid = price < chan_m
            if prev_dir == -1 and st_dir == 1 and price_above_mid and oi_d >= 0:
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
            elif prev_dir == 1 and st_dir == -1 and price_below_mid and oi_d <= 0:
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
            if self.direction == 1 and st_dir == 1:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and st_dir == -1:
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
