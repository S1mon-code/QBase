import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.trend.aroon import aroon
from indicators.volatility.bollinger import bollinger_bands
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV87(TimeSeriesStrategy):
    """
    策略简介：Aroon趋势方向 + Bollinger Band突破确认的4h多头策略。

    使用指标：
    - Aroon(25): Up > Down且Up > 70确认上升趋势
    - Bollinger Bands(20, 2.0): 价格突破中轨确认方向
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Aroon Up > 70 且 Aroon Up > Aroon Down
    - 价格 > Bollinger中轨

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - Aroon Down > Aroon Up（趋势反转）

    优点：Aroon对趋势方向敏感，BB中轨确认趋势强度
    缺点：横盘时Aroon可能反复穿越
    """
    name = "medium_trend_v87"
    warmup = 200
    freq = "4h"

    aroon_period: int = 25        # Optuna: 14-40
    aroon_threshold: float = 70.0 # Optuna: 50-85
    bb_period: int = 20           # Optuna: 15-30
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._aroon_up = None
        self._aroon_down = None
        self._bb_upper = None
        self._bb_mid = None
        self._bb_lower = None
        self._atr = None
        self._avg_volume = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._aroon_up, self._aroon_down, _ = aroon(highs, lows, period=self.aroon_period)
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            closes, period=self.bb_period, std=2.0
        )
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        au = self._aroon_up[i]
        ad = self._aroon_down[i]
        bb_mid = self._bb_mid[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(au) or np.isnan(ad) or np.isnan(bb_mid):
            return

        close_arr = context.get_full_close_array()
        close_price = close_arr[i]

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

        # 2. Tiered profit-taking
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

        # 3. Signal exit: Aroon reversal
        if side == 1 and ad > au:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and au > self.aroon_threshold and au > ad and close_price > bb_mid:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, au, ad):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, au, ad):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if au <= ad:
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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
