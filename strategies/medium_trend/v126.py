import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.ichimoku import ichimoku
from indicators.volume.volume_spike import volume_spike
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV126(TimeSeriesStrategy):
    """
    策略简介：日线Ichimoku云图定方向 + 5min成交量突增入场的多周期策略。

    使用指标：
    - Ichimoku(9,26,52,26) [日线]: 云图判断趋势方向，价格在云上=做多
    - Volume Spike(20, 2.0) [5min]: 成交量突增信号确认买入意愿
    - ATR(14) [5min]: 止损距离计算

    进场条件（做多）：
    - 日线价格在Ichimoku云上方（close > max(senkou_a, senkou_b)）
    - 5min 出现成交量突增（volume_spike = True）且收阳线

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR / 5ATR）
    - 价格跌破云下方

    优点：Ichimoku多维度确认趋势，量价配合提高入场质量
    缺点：Ichimoku参数固定，不适合所有品种周期
    """
    name = "medium_trend_v126"
    freq = "5min"
    warmup = 2000

    ichi_tenkan: int = 9
    ichi_kijun: int = 26
    vol_spike_threshold: float = 2.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._vol_spike = None
        self._atr = None
        self._avg_volume = None
        self._ichi_senkou_a = None
        self._ichi_senkou_b = None
        self._closes_daily = None
        self._daily_map = None

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
        n = len(closes)

        self._vol_spike = volume_spike(volumes, period=20, threshold=self.vol_spike_threshold)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 48  # 5min * 48 = 4h ~ daily
        n_d = n // step
        trim = n_d * step
        closes_d = closes[:trim].reshape(n_d, step)[:, -1]
        highs_d = highs[:trim].reshape(n_d, step).max(axis=1)
        lows_d = lows[:trim].reshape(n_d, step).min(axis=1)

        tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku(
            highs_d, lows_d, closes_d,
            tenkan=self.ichi_tenkan, kijun=self.ichi_kijun, senkou_b=52, displacement=26
        )
        self._ichi_senkou_a = senkou_a
        self._ichi_senkou_b = senkou_b
        self._closes_daily = closes_d
        self._daily_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1),
                                     len(closes_d) - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._daily_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        vs = self._vol_spike[i]
        sa = self._ichi_senkou_a[j]
        sb = self._ichi_senkou_b[j]
        cd = self._closes_daily[j]
        if np.isnan(atr_val) or np.isnan(sa) or np.isnan(sb):
            return

        cloud_top = max(sa, sb)
        cloud_bottom = min(sa, sb)
        above_cloud = cd > cloud_top
        below_cloud = cd < cloud_bottom
        is_spike = (vs == 1) if not np.isnan(vs) else False
        is_green = context.close_raw > context.open_raw

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

        if side == 0 and above_cloud and is_spike and is_green:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and above_cloud):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
