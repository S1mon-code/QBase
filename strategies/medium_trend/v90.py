import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.psar import psar
from indicators.volume.cmf import cmf
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV90(TimeSeriesStrategy):
    """
    策略简介：Parabolic SAR趋势跟踪 + CMF资金流确认的4h多头策略。

    使用指标：
    - Parabolic SAR(0.02, 0.02, 0.2): SAR在价格下方时为上升趋势
    - CMF(20): Chaikin资金流 > 0确认资金流入
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - SAR方向 = 1（价格在SAR上方）
    - CMF > 0（资金净流入）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - SAR方向翻转为-1

    优点：SAR自带加速因子，趋势越强跟踪越紧
    缺点：SAR在震荡市频繁翻转
    """
    name = "medium_trend_v90"
    warmup = 200
    freq = "4h"

    sar_af_start: float = 0.02   # Optuna: 0.01-0.04
    sar_af_step: float = 0.02    # Optuna: 0.01-0.03
    sar_af_max: float = 0.2      # Optuna: 0.1-0.3
    cmf_period: int = 20         # Optuna: 14-30
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._sar = None
        self._sar_dir = None
        self._cmf = None
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

        self._sar, self._sar_dir = psar(
            highs, lows,
            af_start=self.sar_af_start, af_step=self.sar_af_step, af_max=self.sar_af_max
        )
        self._cmf = cmf(highs, lows, closes, volumes, period=self.cmf_period)
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
        sar_dir = self._sar_dir[i]
        cmf_val = self._cmf[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(sar_dir) or np.isnan(cmf_val):
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

        # 3. Signal exit: SAR flip
        if side == 1 and sar_dir == -1:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and sar_dir == 1 and cmf_val > 0:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, sar_dir, cmf_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, sar_dir, cmf_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if sar_dir != 1 or cmf_val <= 0:
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
