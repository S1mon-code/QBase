import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.bollinger import bollinger_bands
from indicators.volatility.ttm_squeeze import ttm_squeeze
from indicators.volume.volume_spike import volume_spike
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV22(TimeSeriesStrategy):
    """
    策略简介：Bollinger Squeeze + 成交量放量突破的做多策略（10min频率）。

    使用指标：
    - Bollinger Bands(20,2): 挤压检测（带宽收窄）+ 突破方向
    - Volume Spike(20,2.0): 放量确认突破有效
    - TTM Squeeze: 辅助挤压确认
    - ATR(14): 止损距离计算

    进场条件（做多）：BB squeeze释放 + 价格突破上轨 + volume spike
    出场条件：ATR追踪止损 / 分层止盈 / 价格跌破BB中轨

    优点：squeeze+放量双重突破确认
    缺点：squeeze条件严格，信号较少
    """
    name = "mt_v22"
    warmup = 1000
    freq = "10min"

    bb_period: int = 20
    bb_std: float = 2.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._bb_upper = None
        self._bb_mid = None
        self._squeeze = None
        self._mom = None
        self._vol_spike = None
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

        self._bb_upper, self._bb_mid, _ = bollinger_bands(closes, period=self.bb_period, std=self.bb_std)
        self._squeeze, self._mom = ttm_squeeze(highs, lows, closes)
        self._vol_spike = volume_spike(volumes, period=20, threshold=2.0)
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
        bb_u = self._bb_upper[i]
        bb_m = self._bb_mid[i]
        squeeze = self._squeeze[i]
        mom = self._mom[i]
        vs = self._vol_spike[i]
        if np.isnan(atr_val) or np.isnan(bb_u) or np.isnan(mom) or np.isnan(vs):
            return
        if i < 1:
            return
        prev_sq = self._squeeze[i - 1]
        if np.isnan(prev_sq):
            return

        self.bars_since_last_scale += 1
        squeeze_fire = (prev_sq == 1) and (squeeze == 0) and mom > 0

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

        if side == 1 and price < bb_m:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and squeeze_fire and price > bb_u and vs == 1:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, mom):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, mom):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if mom <= 0:
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
