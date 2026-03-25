import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.regime.oi_cycle import oi_cycle
from indicators.momentum.stochastic import stochastic
from indicators.seasonality.seasonal_momentum import seasonal_momentum

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV182(TimeSeriesStrategy):
    """
    策略简介：OI Cycle Phase 判断持仓周期 + Stochastic 动量确认 + 季节性动量模式的多空策略。

    使用指标：
    - OI Cycle(60): 检测OI积累/释放周期阶段（0=谷底, 0.5=峰值）
    - Stochastic(14,3): K/D超买超卖判断动量方向
    - Seasonal Momentum(252): 季节性收益率模式，辅助过滤

    进场条件（做多）：OI周期上升阶段(0-0.4)，Stochastic K > D 且 K < 80，季节性动量 > 0
    进场条件（做空）：OI周期下降阶段(0.6-1.0)，Stochastic K < D 且 K > 20，季节性动量 < 0

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Stochastic 反向交叉

    优点：结合持仓周期与季节性规律，多维度过滤假信号
    缺点：OI周期估算依赖lookback长度，季节性模式不稳定
    """
    name = "ag_alltime_v182"
    warmup = 250
    freq = "daily"

    oi_cycle_period: int = 60
    stoch_k: int = 14
    stoch_d: int = 3
    season_lookback: int = 3
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._cycle_phase = None
        self._cycle_amp = None
        self._stoch_k = None
        self._stoch_d = None
        self._seasonal = None
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
        datetimes = context.get_full_datetime_array()

        self._atr = atr(highs, lows, closes, period=14)
        self._cycle_phase, self._cycle_amp = oi_cycle(oi, period=self.oi_cycle_period)
        self._stoch_k, self._stoch_d = stochastic(highs, lows, closes, k=self.stoch_k, d=self.stoch_d)
        self._seasonal = seasonal_momentum(closes, datetimes, lookback_years=self.season_lookback)

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
        phase = self._cycle_phase[i]
        sk = self._stoch_k[i]
        sd = self._stoch_d[i]
        seas = self._seasonal[i]
        if np.isnan(phase) or np.isnan(sk) or np.isnan(sd) or np.isnan(seas):
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
        if side == 1 and sk < sd:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and sk > sd:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            # Long: OI building phase + stoch bullish + seasonal positive
            if phase < 0.4 and sk > sd and sk < 80 and seas > 0:
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
            # Short: OI declining phase + stoch bearish + seasonal negative
            elif phase > 0.6 and sk < sd and sk > 20 and seas < 0:
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
            if self.direction == 1 and sk > sd and phase < 0.5:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and sk < sd and phase > 0.5:
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
