"""Strong Trend v74 — OI Momentum-Price Divergence + ATR Ratio."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()

from indicators.structure.oi_momentum_divergence import oi_momentum_price_divergence
from indicators.volatility.chop import atr_ratio
from indicators.volatility.atr import atr


class StrongTrendV74(TimeSeriesStrategy):
    """
    策略简介：OI Momentum-Price Divergence 持仓动量背离 + ATR Ratio 波动扩张策略。

    使用指标：
    - OI Momentum-Price Divergence(20): OI动量与价格动量的背离
      >0 表示 OI 动量 确认价格方向（非背离，趋势健康）
    - ATR Ratio(5, 20): 短期ATR/长期ATR，>1.2 说明波动扩张（突破状态）
    - ATR(14): 追踪止损

    进场条件（做多）：
    - OI-Price Divergence > 0（OI确认价格趋势，资金追随价格）
    - ATR Ratio > atr_ratio_thresh（波动扩张，趋势启动）

    出场条件：
    - ATR 追踪止损
    - OI-Price Divergence < div_exit_thresh（持仓与价格出现背离）

    优点：背离检测直接识别趋势健康度，ATR Ratio 捕捉波动率突变
    缺点：背离指标在横盘时信号不稳定
    """
    name = "strong_trend_v74"
    warmup = 60
    freq = "daily"

    div_period: int = 20
    atr_ratio_thresh: float = 1.2
    div_exit_thresh: float = -0.5
    atr_trail_mult: float = 4.0

    def __init__(self):
        super().__init__()
        self._divergence = None
        self._atr_ratio = None
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

        self._divergence = oi_momentum_price_divergence(closes, oi,
                                                          period=self.div_period)
        self._atr_ratio = atr_ratio(highs, lows, closes, short=5, long=20)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return

        div_val = self._divergence[i]
        ar = self._atr_ratio[i]
        atr_val = self._atr[i]
        if np.isnan(div_val) or np.isnan(ar) or np.isnan(atr_val):
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

        # Entry: OI confirms price + volatility expanding
        if side == 0 and div_val > 0 and ar > self.atr_ratio_thresh:
            lot_size = self._calc_lots(context, price, atr_val)
            if lot_size > 0:
                context.buy(lot_size)
                self.entry_price = price
                self.highest = price
                self.stop_price = price - self.atr_trail_mult * atr_val

        # Signal exit: divergence emerges
        elif side == 1 and div_val < self.div_exit_thresh:
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
