"""Strong Trend v66 — OI Rate of Change + PPO momentum."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.volume.oi_rate_of_change import oi_roc
from indicators.momentum.ppo import ppo
from indicators.volatility.atr import atr


class StrongTrendV66(TimeSeriesStrategy):
    """
    策略简介：OI Rate of Change 持仓变化率 + PPO 百分比价格振荡策略。

    使用指标：
    - OI ROC(14): 持仓量变化率，>oi_roc_thresh 表示加仓
    - PPO(12, 26, 9): 百分比价格振荡，PPO line > signal 看多
    - ATR(14): 追踪止损

    进场条件（做多）：
    - OI ROC > oi_roc_threshold（持仓量快速增加）
    - PPO line > PPO signal（价格动量正向）

    出场条件：
    - ATR 追踪止损
    - PPO histogram < 0 且 OI ROC < 0（动量 + 持仓双翻空）

    优点：OI ROC 直接反映市场参与热度变化，PPO 标准化便于跨品种
    缺点：OI ROC 在交割月前可能因换仓失真
    """
    name = "strong_trend_v66"
    warmup = 60
    freq = "daily"

    oi_roc_period: int = 14
    oi_roc_threshold: float = 10.0
    ppo_fast: int = 12
    ppo_slow: int = 26
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._oi_roc = None
        self._ppo_line = None
        self._ppo_signal = None
        self._ppo_hist = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.highest = 0.0
        self.stop_price = 0.0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        oi = context.get_full_oi_array()

        self._oi_roc = oi_roc(oi, period=self.oi_roc_period)
        ppo_line, ppo_sig, ppo_hist = ppo(closes, fast=self.ppo_fast,
                                           slow=self.ppo_slow, signal=9)
        self._ppo_line = ppo_line
        self._ppo_signal = ppo_sig
        self._ppo_hist = ppo_hist
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        oi_roc_val = self._oi_roc[i]
        ppo_l = self._ppo_line[i]
        ppo_s = self._ppo_signal[i]
        ppo_h = self._ppo_hist[i]
        atr_val = self._atr[i]
        if np.isnan(oi_roc_val) or np.isnan(ppo_l) or np.isnan(atr_val):
            return

        # Stop loss
        if side == 1:
            self.highest = max(self.highest, price)
            trailing = self.highest - self.atr_trail_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset()
                return

        # Entry: OI expanding + PPO bullish
        if side == 0 and oi_roc_val > self.oi_roc_threshold and ppo_l > ppo_s:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: both turning bearish
        elif side == 1 and ppo_h < 0 and oi_roc_val < 0:
            context.close_long()
            self._reset()

    def _calc_lots(self, context, price, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
        stop_dist = self.atr_trail_mult * atr_val * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = price * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        return max(1, min(risk_lots, int(context.equity * 0.30 / margin)))

    def _reset(self):
        self.entry_price = 0.0
        self.highest = 0.0
        self.stop_price = 0.0
