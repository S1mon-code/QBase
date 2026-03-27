import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.regime.market_state import market_state
from indicators.trend.ema import ema_cross
from indicators.regime.oi_regime import oi_regime

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV181(TimeSeriesStrategy):
    """
    策略简介：Market Phase 状态分类 + EMA 交叉趋势确认 + OI Regime 持仓结构过滤的多空策略。

    使用指标：
    - Market State(20): 分类当前市场阶段（0=安静,1=上升趋势,2=下降趋势,3=震荡,4=突破）
    - EMA Cross(20,60): 快慢EMA交叉确认趋势方向
    - OI Regime(60): 持仓量结构判断资金流入/流出

    进场条件（做多）：market_state=1(上升趋势) 或 4(突破向上)，EMA金叉，OI regime > 0（资金流入）
    进场条件（做空）：market_state=2(下降趋势)，EMA死叉，OI regime < 0（资金流出）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Market state 切换为不利状态

    优点：三维过滤（状态+趋势+结构），在积累/分配阶段精确入场
    缺点：Market state 分类滞后，可能错过快速反转
    """
    name = "ag_alltime_v181"
    warmup = 200
    freq = "daily"

    ms_period: int = 20
    ema_fast: int = 20
    ema_slow: int = 60
    oi_period: int = 60
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._state = None
        self._state_conf = None
        self._ema_fast_arr = None
        self._ema_slow_arr = None
        self._ema_cross_dir = None
        self._oi_regime = None
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
        self._state, self._state_conf = market_state(closes, volumes, oi, period=self.ms_period)
        self._ema_fast_arr, self._ema_slow_arr, self._ema_cross_dir = ema_cross(
            closes, fast_period=self.ema_fast, slow_period=self.ema_slow)
        self._oi_regime = oi_regime(closes, oi, volumes, period=self.oi_period)

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
        cross_val = self._ema_cross_dir[i]
        oi_reg = self._oi_regime[i]
        if np.isnan(state_val) or np.isnan(cross_val) or np.isnan(oi_reg):
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
        if side == 1 and (state_val == 2 or cross_val < 0):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (state_val == 1 or state_val == 4 or cross_val > 0):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            # Long: accumulation/breakout + EMA golden cross + OI inflow
            if state_val in (1, 4) and cross_val > 0 and oi_reg > 0:
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
            # Short: distribution + EMA death cross + OI outflow
            elif state_val == 2 and cross_val < 0 and oi_reg < 0:
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
            if self.direction == 1 and cross_val > 0 and state_val in (1, 4):
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and cross_val < 0 and state_val == 2:
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
