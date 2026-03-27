import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.momentum.tsi import tsi
from indicators.trend.aroon import aroon
from indicators.spread.gold_silver_ratio import gold_silver_ratio
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV191(TimeSeriesStrategy):
    """
    策略简介：TSI零轴交叉 + Aroon趋势确认 + 金银比有利区间的跨品种趋势策略。

    使用指标：
    - TSI(25,13): 真实强度指标，零轴交叉确认动量方向
    - Aroon(25): 趋势方向和强度，Up>Down=上升趋势
    - Gold/Silver Ratio(60): 金银比z-score，高z-score=白银相对低估
    - ATR(14): 止损距离计算

    进场条件（做多）：TSI从负穿零轴以上 + Aroon Up>Down + 金银比z-score>0
    进场条件（做空）：TSI从正穿零轴以下 + Aroon Down>Up + 金银比z-score<0

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - TSI反向穿越零轴

    优点：跨品种信号提供独立视角，TSI平滑减少噪音
    缺点：金银比变化缓慢可能降低信号频率
    """
    name = "ag_alltime_v191"
    warmup = 200
    freq = "daily"

    tsi_long: int = 25            # Optuna: 20-30
    tsi_short: int = 13           # Optuna: 10-16
    aroon_period: int = 25        # Optuna: 14-30
    gsr_period: int = 60          # Optuna: 40-80
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._tsi = None
        self._aroon_up = None
        self._aroon_down = None
        self._gsr_ratio = None
        self._gsr_zscore = None
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

        # Load AU closes for gold/silver ratio
        au_closes = context.load_auxiliary_close("AU")

        self._tsi, self._tsi_sig = tsi(closes, self.tsi_long, self.tsi_short)
        self._aroon_up, self._aroon_down, _ = aroon(highs, lows, self.aroon_period)
        self._gsr_ratio, self._gsr_zscore = gold_silver_ratio(
            au_closes, closes, self.gsr_period)
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
        tsi_val = self._tsi[i]
        aroon_u = self._aroon_up[i]
        aroon_d = self._aroon_down[i]
        gsr_z = self._gsr_zscore[i]

        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(tsi_val):
            return
        if np.isnan(aroon_u) or np.isnan(aroon_d):
            return
        if np.isnan(gsr_z):
            gsr_z = 0.0

        prev_tsi = self._tsi[i - 1] if i > 0 else np.nan
        if np.isnan(prev_tsi):
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

        # 3. Signal exit: TSI crosses zero in opposite direction
        if side == 1 and tsi_val < 0 and prev_tsi >= 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and tsi_val > 0 and prev_tsi <= 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: TSI zero cross + Aroon trend + favorable Au/Ag ratio
        if side == 0:
            if prev_tsi <= 0 and tsi_val > 0 and aroon_u > aroon_d and gsr_z > 0:
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
            elif prev_tsi >= 0 and tsi_val < 0 and aroon_d > aroon_u and gsr_z < 0:
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
            if self.direction == 1 and tsi_val > 0 and aroon_u > aroon_d:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and tsi_val < 0 and aroon_d > aroon_u:
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
