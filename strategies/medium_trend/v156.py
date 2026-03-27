import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.ichimoku import ichimoku
from indicators.trend.ema import ema
from indicators.momentum.connors_rsi import connors_rsi
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV156(TimeSeriesStrategy):
    """
    策略简介：日线Ichimoku + 4h EMA方向 + 5min Connors RSI三周期策略。

    使用指标：
    - Ichimoku(9,26,52,26) [日线]: 云上确认大趋势
    - EMA(20) [4h]: close > EMA确认中周期
    - Connors RSI(3,2,100) [5min]: CRSI < 15超卖入场
    - ATR(14) [5min]: 止损距离

    进场条件（做多）：日线云上 + 4h close>EMA + 5min CRSI<15
    出场条件：ATR追踪止损, 分层止盈, 云下平仓

    优点：Connors RSI比传统RSI更灵敏
    缺点：CRSI极端值持续时间短，需快速反应
    """
    name = "medium_trend_v156"
    freq = "5min"
    warmup = 3000

    crsi_entry: float = 15.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._crsi = None
        self._atr = None
        self._avg_volume = None
        self._senkou_a = None
        self._senkou_b = None
        self._closes_d = None
        self._ema_4h = None
        self._closes_4h = None
        self._d_map = None

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

        self._crsi = connors_rsi(closes, rsi=3, streak=2, pct_rank=100)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 48
        nd = n // step
        trim = nd * step
        closes_d = closes[:trim].reshape(nd, step)[:, -1]
        highs_d = highs[:trim].reshape(nd, step).max(axis=1)
        lows_d = lows[:trim].reshape(nd, step).min(axis=1)

        _, _, sa, sb, _ = ichimoku(highs_d, lows_d, closes_d, tenkan=9, kijun=26)
        self._senkou_a = sa
        self._senkou_b = sb
        self._closes_d = closes_d
        self._ema_4h = ema(closes_d, period=20)
        self._closes_4h = closes_d
        self._d_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1), nd - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._d_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        crsi_val = self._crsi[i]
        atr_val = self._atr[i]
        sa = self._senkou_a[j]
        sb = self._senkou_b[j]
        cd = self._closes_d[j]
        ema_val = self._ema_4h[j]
        if np.isnan(crsi_val) or np.isnan(atr_val) or np.isnan(sa) or np.isnan(sb) or np.isnan(ema_val):
            return

        cloud_top = max(sa, sb)
        above_cloud = cd > cloud_top
        above_ema = cd > ema_val
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

        if side == 1 and not above_cloud:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and above_cloud and above_ema and crsi_val < self.crsi_entry:
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
                    and above_cloud and above_ema):
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
