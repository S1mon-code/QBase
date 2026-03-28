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
from indicators.structure.volume_oi_ratio import volume_oi_ratio
from indicators.trend.ema import ema

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class MediumTrendV208(TimeSeriesStrategy):
    """
    策略简介：Volume/OI Ratio活跃度 + EMA交叉趋势的日线做多策略。

    使用指标：
    - Volume/OI Ratio(20): 成交量/持仓量比，反映市场活跃度
    - EMA Cross(10, 30): 快慢均线交叉判定趋势方向
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Volume/OI Ratio > 1.2（市场活跃度提升）
    - EMA(10) > EMA(30)（短期均线在上，多头排列）

    出场条件：
    - ATR追踪止损（3.5倍ATR）
    - 分层止盈（3ATR/5ATR）
    - EMA(10) < EMA(30)（均线死叉）

    优点：Vol/OI比率筛选活跃市场，避免低流动性陷阱
    缺点：OI数据延迟可能影响信号及时性
    """
    name = "mt_v208"
    warmup = 60
    freq = "daily"

    voi_period: int = 20
    voi_threshold: float = 1.2
    ema_fast: int = 10
    ema_slow: int = 30
    atr_stop_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._voi = None
        self._ema_cross = None
        self._ema_f = None
        self._ema_s = None
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
        oi = context.get_full_oi_array()
        self._atr = atr(highs, lows, closes, period=14)
        self._voi = volume_oi_ratio(volumes, oi, period=self.voi_period)
        self._ema_f = ema(closes, self.ema_fast)
        self._ema_s = ema(closes, self.ema_slow)
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
        voi_val = self._voi[i]
        ef = self._ema_f[i]
        es = self._ema_s[i]
        if np.isnan(voi_val) or np.isnan(ef) or np.isnan(es):
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

        if side == 1 and ef < es:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and voi_val > self.voi_threshold and ef > es:
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
