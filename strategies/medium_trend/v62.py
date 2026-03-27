import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.trend.ichimoku import ichimoku
from indicators.volume.cmf import cmf
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV62(TimeSeriesStrategy):
    """
    策略简介：Ichimoku Cloud趋势判断 + CMF资金流确认的1h策略。

    使用指标：
    - Ichimoku(9,26,52,26): 价格在云层上方且转换>基准为多头
    - CMF(20): 正值为资金流入
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - close > Senkou Span A and Senkou Span B（价格在云层上方）
    - 转换线 > 基准线（短期趋势上行）
    - CMF > 0（资金流入）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - close < 云层下沿（跌入云层）

    优点：Ichimoku提供多维度趋势信息，CMF量化资金流
    缺点：Ichimoku参数多，在低波动品种信号稀少
    """
    name = "medium_trend_v62"
    warmup = 500
    freq = "1h"

    ichi_tenkan: int = 9
    ichi_kijun: int = 26
    ichi_senkou: int = 52
    ichi_offset: int = 26
    cmf_period: int = 20
    atr_stop_mult: float = 3.0       # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._tenkan = None
        self._kijun = None
        self._senkou_a = None
        self._senkou_b = None
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

        self._tenkan, self._kijun, self._senkou_a, self._senkou_b, _ = ichimoku(
            highs, lows, closes, self.ichi_tenkan, self.ichi_kijun, self.ichi_senkou, self.ichi_offset)
        self._cmf = cmf(highs, lows, closes, volumes, period=self.cmf_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        close_val = context.get_full_close_array()[i]
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        tenkan = self._tenkan[i]
        kijun = self._kijun[i]
        sa = self._senkou_a[i]
        sb = self._senkou_b[i]
        cmf_val = self._cmf[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(sa) or np.isnan(sb) or np.isnan(cmf_val):
            return
        if np.isnan(tenkan) or np.isnan(kijun):
            return

        cloud_top = max(sa, sb)
        cloud_bottom = min(sa, sb)
        above_cloud = close_val > cloud_top
        below_cloud = close_val < cloud_bottom
        tenkan_above = tenkan > kijun

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

        if side == 1 and below_cloud:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and above_cloud and tenkan_above and cmf_val > 0:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, above_cloud, tenkan_above):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, above_cloud, tenkan_above):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if not above_cloud or not tenkan_above:
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
