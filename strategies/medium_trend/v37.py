import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.regime.composite_regime_score import composite_regime
from indicators.momentum.stoch_rsi import stoch_rsi
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV37(TimeSeriesStrategy):
    """
    策略简介：Composite Regime Score行情判断 + Stochastic RSI精确入场的做多策略（10min频率）。

    使用指标：
    - Composite Regime Score(20): 综合行情评分（趋势/震荡/突破）
    - Stochastic RSI(14,14,3,3): 超卖区入场
    - ATR(14): 止损距离计算

    进场条件（做多）：regime score > 0.5（趋势行情） 且 StochRSI %K < 20（超卖）
    出场条件：ATR追踪止损 / 分层止盈 / StochRSI %K > 80

    优点：综合评分过滤非趋势行情，StochRSI精确入场
    缺点：综合评分计算多指标可能较慢
    """
    name = "mt_v37"
    warmup = 1000
    freq = "10min"

    regime_period: int = 20
    regime_threshold: float = 0.5
    srsi_rsi: int = 14
    srsi_stoch: int = 14
    srsi_k: int = 3
    srsi_d: int = 3
    srsi_entry: float = 20.0
    srsi_exit: float = 80.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._regime = None
        self._srsi_k = None
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

        self._regime = composite_regime(closes, highs, lows, period=self.regime_period)
        sk, _ = stoch_rsi(closes, rsi=self.srsi_rsi, stoch=self.srsi_stoch,
                          k=self.srsi_k, d=self.srsi_d)
        self._srsi_k = sk
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
        reg = self._regime[i]
        sk = self._srsi_k[i]
        if np.isnan(atr_val) or np.isnan(reg) or np.isnan(sk):
            return

        self.bars_since_last_scale += 1
        is_trend = reg > self.regime_threshold

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

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

        if side == 1 and sk > self.srsi_exit:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and is_trend and sk < self.srsi_entry:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, is_trend):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, is_trend):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if not is_trend:
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
