import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager
from indicators.volatility.atr import atr
from indicators.volatility.bollinger import bollinger_bands

_SPEC_MANAGER = ContractSpecManager()

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _gold_silver_ratio(ag_closes, au_closes, period=60):
    """Gold/Silver ratio and its z-score for mean-reversion trading.

    ratio = AU_price / AG_price
    When ratio is high (z > 1.5) = silver cheap, go long silver
    When ratio is low (z < -1.5) = silver expensive, go short silver
    """
    n = len(ag_closes)
    ratio = np.full(n, np.nan, dtype=np.float64)
    ratio_z = np.full(n, np.nan, dtype=np.float64)

    min_len = min(len(ag_closes), len(au_closes))
    for i in range(min_len):
        if ag_closes[i] > 0:
            ratio[i] = au_closes[i] / ag_closes[i]

    for i in range(period, n):
        window = ratio[i - period + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid)
        if sigma < 1e-12:
            ratio_z[i] = 0.0
        else:
            ratio_z[i] = (ratio[i] - mu) / sigma

    return ratio, ratio_z


class StrategyV14(TimeSeriesStrategy):
    """
    策略简介：基于金银比价均值回归的白银多空策略。

    使用指标：
    - Gold/Silver Ratio(60): 金银价格比及其z-score
    - Bollinger Bands(20): 银价自身的超买超卖
    - ATR(14): 止损距离计算

    进场条件（做多）：金银比z-score>1.5（银便宜），银价<BB中轨
    进场条件（做空）：金银比z-score<-1.5（银贵），银价>BB中轨

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 金银比z-score回到[-0.5, 0.5]

    优点：跨品种基本面逻辑支撑，金银比长期稳定均值回归
    缺点：需要AU数据同步，极端行情下比价可能长期偏离
    """
    name = "ag_alltime_v14"
    warmup = 250
    freq = "daily"

    ratio_period: int = 60        # Optuna: 30-120
    entry_z: float = 1.5         # Optuna: 1.0-2.5
    exit_z: float = 0.5          # Optuna: 0.2-1.0
    bb_period: int = 20           # Optuna: 15-30
    atr_stop_mult: float = 3.5   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._ratio = None
        self._ratio_z = None
        self._bb_upper = None
        self._bb_mid = None
        self._bb_lower = None
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
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            closes, period=self.bb_period
        )

        # Load auxiliary gold data
        au_closes = context.load_auxiliary_close("AU")
        self._ratio, self._ratio_z = _gold_silver_ratio(
            closes, au_closes, period=self.ratio_period
        )

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        rz = self._ratio_z[i]
        bb_mid = self._bb_mid[i]
        if np.isnan(rz) or np.isnan(bb_mid):
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

        # 3. Signal-based exit: ratio reverted to neutral
        if side == 1 and rz < self.exit_z:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and rz > -self.exit_z:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if rz > self.entry_z and price < bb_mid:
                # Ratio high = silver cheap, go long silver
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
            elif rz < -self.entry_z and price > bb_mid:
                # Ratio low = silver expensive, go short silver
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
            if self.direction == 1 and rz > self.entry_z * 1.2:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and rz < -self.entry_z * 1.2:
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
        spec = _SPEC_MANAGER.get(context.symbol)
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
