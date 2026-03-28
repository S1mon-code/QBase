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
from indicators.microstructure.amihud import amihud_illiquidity
from indicators.volume.volume_spike import volume_spike

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class MediumTrendV250(TimeSeriesStrategy):
    """
    策略简介：Amihud低流动性突破 + Volume Spike放量确认的5min做多策略。

    使用指标：
    - Amihud Illiquidity(20): 低流动性检测，高值=流动性枯竭
    - Volume Spike(20, 2.0): 成交量异常放大检测
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Amihud从高位回落（流动性改善，之前枯竭状态）
    - Volume Spike触发（伴随成交量放大）
    - 价格上涨（方向确认）

    出场条件：
    - ATR追踪止损（3.5倍ATR）
    - 分层止盈
    - Amihud重新飙升（流动性恶化）

    优点：流动性改善+放量=机构入场信号
    缺点：Amihud在极低量时不稳定
    """
    name = "mt_v250"
    warmup = 60
    freq = "5min"

    atr_stop_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._amihud = None
        self._vspike = None
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
        self._amihud = amihud_illiquidity(closes, volumes)
        self._vspike = volume_spike(volumes)
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
        ami = self._amihud[i]
        vs = self._vspike[i]
        if np.isnan(ami) or np.isnan(vs):
            return
        prev_ami = self._amihud[i - 1] if i > 0 else np.nan
        prev_price = context.get_full_close_array()[i - 1] if i > 0 else np.nan
        if np.isnan(prev_ami) or np.isnan(prev_price):
            return
        self.bars_since_last_scale += 1

        # Amihud declining = liquidity improving
        ami_improving = ami < prev_ami
        price_up = price > prev_price

        # Amihud spike check
        ami_high = False
        if i >= 20:
            ami_win = self._amihud[i - 20:i]
            valid = ami_win[~np.isnan(ami_win)]
            if len(valid) > 0:
                ami_high = ami > np.percentile(valid, 80)

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

        if side == 1 and ami_high:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and ami_improving and vs > 0 and price_up:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0
        elif side == 1 and self.position_scale < MAX_SCALE and self.bars_since_last_scale >= 10 and price > self.entry_price + atr_val:
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
