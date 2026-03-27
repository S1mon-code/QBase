import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.regime.vol_regime_markov import vol_regime_simple
from indicators.volatility.bollinger import bollinger_bands
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class VolRegimeBollinger(TimeSeriesStrategy):
    """
    策略简介：波动率状态 + 布林带的自适应策略。低波时交易布林突破，高波时均值回归。

    使用指标：
    - vol_regime_simple(60): 区分高/低波动率状态
    - bollinger_bands(20, 2.0): 布林带上下轨 + 中轨
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - 低波状态: 价格突破布林上轨（突破做多）
    - 高波状态: 价格触及布林下轨（均值回归做多）

    进场条件（做空）：
    - 低波状态: 价格跌破布林下轨（突破做空）
    - 高波状态: 价格触及布林上轨（均值回归做空）

    出场条件：
    - ATR 追踪止损 / 分层止盈
    - 低波突破: 价格回穿中轨退出
    - 高波均值回归: 价格回到中轨退出

    优点：自适应不同波动率环境，避免在高波中追突破
    缺点：状态切换时可能产生虚假信号
    """
    name = "v61_vol_regime_bollinger"
    warmup = 200
    freq = "4h"

    regime_period: int = 60
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._regime = None
        self._bb_upper = None
        self._bb_middle = None
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
        self.entry_regime = -1  # 0=low vol, 1=high vol

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._regime, _, _ = vol_regime_simple(closes, period=self.regime_period)
        self._bb_upper, self._bb_middle, self._bb_lower = bollinger_bands(
            closes, period=self.bb_period, num_std=self.bb_std)
        self._atr = atr(highs, lows, closes, period=self.atr_period)

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        regime_val = self._regime[i]
        bb_upper = self._bb_upper[i]
        bb_middle = self._bb_middle[i]
        bb_lower = self._bb_lower[i]
        atr_val = self._atr[i]
        if np.isnan(regime_val) or np.isnan(bb_upper) or np.isnan(atr_val):
            return

        is_low_vol = (regime_val == 0)
        self.bars_since_last_scale += 1

        # ── 1. 止损检查 ──
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

        # ── 2. 分层止盈 ──
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
        elif side == -1 and self.entry_price > 0:
            profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_3atr = True
                return

        # ── 3. 信号弱化退出 ──
        if side == 1:
            if is_low_vol and price < bb_middle:
                context.close_long()
                self._reset_state()
                return
            if not is_low_vol and price >= bb_middle:
                context.close_long()
                self._reset_state()
                return
        elif side == -1:
            if is_low_vol and price > bb_middle:
                context.close_short()
                self._reset_state()
                return
            if not is_low_vol and price <= bb_middle:
                context.close_short()
                self._reset_state()
                return

        # ── 4. 入场逻辑 ──
        if side == 0:
            if is_low_vol:
                if price > bb_upper:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.buy(base_lots)
                        self._set_entry(price, price - self.atr_stop_mult * atr_val, 0)
                elif price < bb_lower:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.sell(base_lots)
                        self._set_entry(price, price + self.atr_stop_mult * atr_val, 0)
            else:
                if price <= bb_lower:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.buy(base_lots)
                        self._set_entry(price, price - self.atr_stop_mult * atr_val, 1)
                elif price >= bb_upper:
                    base_lots = self._calc_lots(context, atr_val)
                    if base_lots > 0:
                        context.sell(base_lots)
                        self._set_entry(price, price + self.atr_stop_mult * atr_val, 1)

        # ── 5. 加仓逻辑 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.sell(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _set_entry(self, price, stop, regime):
        self.entry_price = price
        self.stop_price = stop
        self.highest_since_entry = price
        self.lowest_since_entry = price
        self.position_scale = 1
        self.bars_since_last_scale = 0
        self.entry_regime = regime

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
        self.entry_regime = -1
