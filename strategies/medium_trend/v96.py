import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.tema import tema
from indicators.volatility.chandelier_exit import chandelier_exit
from indicators.volatility.atr import atr
from indicators.volume.volume_oscillator import volume_oscillator

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV96(TimeSeriesStrategy):
    """
    策略简介：TEMA方向判断 + Chandelier Exit止损 + Volume Oscillator确认的4h多头策略。

    使用指标：
    - TEMA(20): Triple EMA方向判断，上升为多头
    - Chandelier Exit(22, 3.0): 自带波动率自适应止损
    - Volume Oscillator(5, 20): > 0确认成交量放大
    - ATR(14): 辅助止损

    进场条件（做多）：
    - TEMA上升
    - 价格 > Chandelier Exit长线
    - Volume Oscillator > 0

    出场条件：
    - Chandelier Exit触发或ATR追踪止损（取更紧的）
    - 分层止盈（3ATR/5ATR）
    - TEMA下降

    优点：TEMA响应极快，Chandelier自适应止损
    缺点：TEMA容易被短期波动误导
    """
    name = "medium_trend_v96"
    warmup = 200
    freq = "4h"

    tema_period: int = 20         # Optuna: 12-30
    chand_period: int = 22        # Optuna: 14-30
    chand_mult: float = 3.0       # Optuna: 2.0-4.0
    atr_stop_mult: float = 3.5    # Optuna: 2.5-5.0

    def __init__(self):
        super().__init__()
        self._tema = None
        self._chand_long = None
        self._chand_short = None
        self._vol_osc = None
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

        self._tema = tema(closes, self.tema_period)
        self._chand_long, self._chand_short = chandelier_exit(
            highs, lows, closes, period=self.chand_period, mult=self.chand_mult
        )
        self._vol_osc = volume_oscillator(volumes, fast=5, slow=20)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        tema_val = self._tema[i]
        chand_l = self._chand_long[i]
        vosc = self._vol_osc[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(tema_val) or np.isnan(chand_l) or np.isnan(vosc):
            return
        tema_prev = self._tema[i - 2] if i >= 2 else np.nan
        if np.isnan(tema_prev):
            return

        closes = context.get_full_close_array()
        close_price = closes[i]
        tema_rising = tema_val > tema_prev

        self.bars_since_last_scale += 1

        # 1. Stop loss (ATR trailing or Chandelier, whichever tighter)
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            atr_trail = self.highest_since_entry - self.atr_stop_mult * atr_val
            stop_level = max(self.stop_price, atr_trail, chand_l)
            self.stop_price = stop_level
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

        # 3. Signal exit: TEMA declining
        if side == 1 and not tema_rising:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and tema_rising and close_price > chand_l and vosc > 0:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = max(price - self.atr_stop_mult * atr_val, chand_l)
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, tema_rising):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, tema_rising):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if not tema_rising:
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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
