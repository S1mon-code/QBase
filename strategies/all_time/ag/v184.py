import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.regime.momentum_regime import momentum_regime
from indicators.momentum.roc import rate_of_change
from indicators.spread.cross_momentum import cross_momentum

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV184(TimeSeriesStrategy):
    """
    策略简介：Momentum Regime 群体行为检测 + ROC 动量 + 跨品种动量确认的多空策略。

    使用指标：
    - Momentum Regime(10,60): 快慢动量判断当前是趋势还是震荡
    - ROC(20): 价格变化率，确认方向和动量强度
    - Cross Momentum(AG vs AU, 20): 跨品种动量确认，金银联动

    进场条件（做多）：momentum regime 趋势偏多 + ROC > 0 + 跨品种动量正
    进场条件（做空）：momentum regime 趋势偏空 + ROC < 0 + 跨品种动量负

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - ROC 反向

    优点：群体行为 + 跨市场确认，避免单品种假信号
    缺点：需要AU辅助数据，金银相关性下降时失效
    """
    name = "ag_alltime_v184"
    warmup = 500
    freq = "4h"

    mom_fast: int = 10
    mom_slow: int = 60
    roc_period: int = 20
    cross_mom_period: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._mom_regime = None
        self._roc = None
        self._cross_mom = None
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
        self._mom_regime = momentum_regime(closes, fast=self.mom_fast, slow=self.mom_slow)
        self._roc = rate_of_change(closes, self.roc_period)

        # Cross momentum with AU
        au_closes = context.load_auxiliary_close("AU")
        if au_closes is not None and len(au_closes) == len(closes):
            self._cross_mom = cross_momentum(closes, au_closes, period=self.cross_mom_period)
        else:
            self._cross_mom = np.full(len(closes), np.nan)

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
        regime_val = self._mom_regime[i]
        roc_val = self._roc[i]
        cm_val = self._cross_mom[i]
        if np.isnan(regime_val) or np.isnan(roc_val) or np.isnan(cm_val):
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
        if side == 1 and roc_val < 0:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and roc_val > 0:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry: regime aligned + ROC direction + cross momentum
        # momentum_regime returns a value; >0 bullish trend, <0 bearish trend
        if side == 0:
            if regime_val > 0 and roc_val > 0 and cm_val > 0:
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
            elif regime_val < 0 and roc_val < 0 and cm_val < 0:
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
            if self.direction == 1 and roc_val > 0 and regime_val > 0:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and roc_val < 0 and regime_val < 0:
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
