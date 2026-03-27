import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.volume.oi_momentum import oi_momentum
from indicators.trend.ema import ema

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _oi_regime(oi_arr, closes, period=20):
    """OI regime classification based on OI expansion/contraction + price direction.

    Returns regime array:
    1 = OI expanding + price up (new longs, strong bullish)
    2 = OI expanding + price down (new shorts, strong bearish)
    3 = OI contracting + price up (short covering, weak bullish)
    4 = OI contracting + price down (long liquidation, weak bearish)
    0 = neutral
    """
    n = len(closes)
    regime = np.full(n, 0, dtype=np.int32)
    strength = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return regime, strength

    for i in range(period, n):
        oi_chg = (oi_arr[i] - oi_arr[i - period]) / max(oi_arr[i - period], 1.0)
        price_chg = (closes[i] - closes[i - period]) / closes[i - period]

        # Expansion/contraction threshold
        oi_expanding = oi_chg > 0.05
        oi_contracting = oi_chg < -0.05
        price_up = price_chg > 0
        price_down = price_chg < 0

        if oi_expanding and price_up:
            regime[i] = 1
            strength[i] = min(1.0, abs(oi_chg) * 10)
        elif oi_expanding and price_down:
            regime[i] = 2
            strength[i] = min(1.0, abs(oi_chg) * 10)
        elif oi_contracting and price_up:
            regime[i] = 3
            strength[i] = min(1.0, abs(oi_chg) * 5)
        elif oi_contracting and price_down:
            regime[i] = 4
            strength[i] = min(1.0, abs(oi_chg) * 5)
        else:
            regime[i] = 0
            strength[i] = 0.0

    return regime, strength


class StrategyV10(TimeSeriesStrategy):
    """
    策略简介：基于OI持仓量体制变化和价格方向的交易策略。

    使用指标：
    - OI Regime: 持仓量扩张/收缩 + 价格方向分类
    - OI Momentum(20): 持仓量变化率
    - EMA(30): 趋势过滤
    - ATR(14): 止损距离计算

    进场条件（做多）：OI regime=1(OI扩张+价格涨)，OI动量>0.05，价格>EMA
    进场条件（做空）：OI regime=2(OI扩张+价格跌)，OI动量>0.05，价格<EMA

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - OI regime切换为弱势或反向

    优点：利用期货特有OI信息，区分新建仓和平仓行为
    缺点：OI数据质量依赖交易所，部分品种OI波动大
    """
    name = "ag_alltime_v10"
    warmup = 200
    freq = "daily"

    oi_period: int = 20           # Optuna: 10-40
    ema_period: int = 30          # Optuna: 15-50
    oi_mom_thresh: float = 0.05  # Optuna: 0.02-0.10
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._oi_regime = None
        self._oi_strength = None
        self._oi_mom = None
        self._ema = None
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
        self.direction = 0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi_arr = context.get_full_oi_array()

        self._atr = atr(highs, lows, closes, period=14)
        self._ema = ema(closes, period=self.ema_period)
        self._oi_mom = oi_momentum(oi_arr, period=self.oi_period)
        self._oi_regime, self._oi_strength = _oi_regime(oi_arr, closes, period=self.oi_period)

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        regime = self._oi_regime[i]
        oi_str = self._oi_strength[i]
        oi_mom = self._oi_mom[i]
        ema_val = self._ema[i]
        if np.isnan(oi_str) or np.isnan(oi_mom) or np.isnan(ema_val):
            return

        self.bars_since_last_scale += 1

        # 1. Stop loss
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

        # 2. Tiered profit-taking
        if side != 0 and self.entry_price > 0:
            if side == 1:
                profit_atr = (price - self.entry_price) / atr_val
            else:
                profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_3atr = True
                return

        # 3. Signal-based exit: regime shift to weak or opposite
        if side == 1 and regime in (2, 4):  # bearish regimes
            context.close_long()
            self._reset_state()
            return
        if side == -1 and regime in (1, 3):  # bullish regimes
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0:
            if regime == 1 and oi_mom > self.oi_mom_thresh and price > ema_val:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
                    self.direction = 1
            elif regime == 2 and oi_mom > self.oi_mom_thresh and price < ema_val:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
                    self.direction = -1

        # 5. Scale-in
        elif side != 0 and self._should_add(price, atr_val):
            if self.direction == 1 and regime == 1:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and regime == 2:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.sell(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1 and price < self.entry_price + atr_val:
            return False
        if self.direction == -1 and price > self.entry_price - atr_val:
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
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.direction = 0
