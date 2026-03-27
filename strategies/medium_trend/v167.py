import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.volatility.atr import atr
from indicators.ml.boosting_signal import gradient_boost_signal
from indicators.regime.vol_regime_markov import vol_regime_simple
from indicators.volatility.bollinger import bollinger_bands

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV167(TimeSeriesStrategy):
    """
    策略简介：Gradient Boosting方向预测 + Vol Regime过滤 + Bollinger确认的做多策略（4h）。

    使用指标：
    - Gradient Boost Signal(120): GBT预测正收益概率
    - Vol Regime Simple(60): 高/低波动率regime
    - Bollinger Bands(20,2): 价格在中轨以上确认趋势
    - ATR(14): 止损距离计算

    进场条件（做多）：GBT>0.6 + 低vol regime + 价格>BB中轨
    出场条件：ATR追踪止损 / 分层止盈 / GBT<0.4

    优点：GBT非线性捕获模式，vol regime过滤噪音
    缺点：GBT过拟合风险，低vol可能错过加速期
    """
    name = "mt_v167"
    warmup = 400
    freq = "4h"

    gbt_threshold: float = 0.6
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._gbt = None
        self._vol_regime = None
        self._bb_mid = None
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

        self._atr = atr(highs, lows, closes, period=14)
        from indicators.momentum.rsi import rsi
        features = np.column_stack([rsi(closes, 14), self._atr, closes])
        self._gbt, _ = gradient_boost_signal(closes, features, period=120)
        self._vol_regime, _, _ = vol_regime_simple(closes, period=60)
        _, self._bb_mid, _ = bollinger_bands(closes, period=20, num_std=2.0)
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
        if np.isnan(atr_val) or atr_val <= 0:
            return
        gbt = self._gbt[i]
        vreg = self._vol_regime[i]
        bb_mid = self._bb_mid[i]
        if np.isnan(gbt) or np.isnan(vreg) or np.isnan(bb_mid):
            return
        self.bars_since_last_scale += 1
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
        if side == 1 and gbt < 0.4:
            context.close_long()
            self._reset_state()
            return
        if side == 0 and gbt > self.gbt_threshold and vreg == 0 and price > bb_mid:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0
        elif side == 1 and self.position_scale < MAX_SCALE and self.bars_since_last_scale >= 10 and price > self.entry_price + atr_val and gbt > 0.5:
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
