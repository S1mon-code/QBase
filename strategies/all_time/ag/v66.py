import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.regime.oi_cycle import oi_cycle
from indicators.seasonality.monthly_pattern import monthly_seasonal
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class CyclePhaseMonthly(TimeSeriesStrategy):
    """
    策略简介：OI 周期阶段 + 月度季节性的双重过滤策略。

    使用指标：
    - oi_cycle(60): OI 周期阶段 (0=trough, 0.5=peak) 和振幅
    - monthly_seasonal(lookback_years=3): 月度季节性得分
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - OI 周期阶段从低谷上升 (phase < 0.3, amplitude 上升)
    - 月度季节性得分 > 0（历史同月看涨）

    进场条件（做空）：
    - OI 周期阶段从峰值下降 (phase > 0.7, amplitude 下降)
    - 月度季节性得分 < 0（历史同月看跌）

    出场条件：
    - ATR 追踪止损 / 分层止盈
    - 周期阶段和季节性信号不再一致

    优点：周期+季节性双重确认，降低假信号
    缺点：季节性规律可能失效，OI 周期不规则
    """
    name = "v66_cycle_phase_monthly"
    warmup = 250
    freq = "daily"

    oi_cycle_period: int = 60
    seasonal_lookback: int = 3
    seasonal_threshold: float = 0.1
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._cycle_phase = None
        self._cycle_amp = None
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

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()
        datetimes = context.get_full_datetime_array()

        self._cycle_phase, self._cycle_amp = oi_cycle(oi, period=self.oi_cycle_period)
        self._seasonal = monthly_seasonal(closes, datetimes,
                                           lookback_years=self.seasonal_lookback)
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

        phase = self._cycle_phase[i]
        amp = self._cycle_amp[i]
        seasonal = self._seasonal[i]
        atr_val = self._atr[i]
        if np.isnan(phase) or np.isnan(seasonal) or np.isnan(atr_val):
            return

        cycle_rising = phase < 0.3 and amp > 0.3
        cycle_falling = phase > 0.7 and amp > 0.3
        seasonal_bull = seasonal > self.seasonal_threshold
        seasonal_bear = seasonal < -self.seasonal_threshold

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
        if side == 1 and (phase > 0.6 or seasonal_bear):
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and (phase < 0.4 or seasonal_bull):
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场逻辑 ──
        if side == 0:
            if cycle_rising and seasonal_bull:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self._set_entry(price, price - self.atr_stop_mult * atr_val)
            elif cycle_falling and seasonal_bear:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self._set_entry(price, price + self.atr_stop_mult * atr_val)

        # ── 5. 加仓逻辑 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and seasonal_bull and phase < 0.5):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and seasonal_bear and phase > 0.5):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.sell(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _set_entry(self, price, stop):
        self.entry_price = price
        self.stop_price = stop
        self.highest_since_entry = price
        self.lowest_since_entry = price
        self.position_scale = 1
        self.bars_since_last_scale = 0

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
