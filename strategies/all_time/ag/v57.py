import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.copula_tail import tail_dependence
from indicators.spread.cross_momentum import cross_momentum
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class CopulaTailCrossMomentumStrategy(TimeSeriesStrategy):
    """
    策略简介：Copula尾部依赖变化 + 跨品种动量确认的多空策略。

    使用指标：
    - Tail Dependence: AG与AU的尾部依赖系数，检测极端共动变化
    - Cross Momentum: AG相对AU的动量差（ROC_AG - ROC_AU）
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - 上尾依赖上升（同涨概率增加）且 cross momentum > 0（AG跑赢AU）

    进场条件（做空）：
    - 下尾依赖上升（同跌概率增加）且 cross momentum < 0（AG跑输AU）

    出场条件：
    - ATR追踪止损 / 分层止盈 / cross momentum反转

    优点：尾部依赖捕捉极端事件中的联动变化，cross momentum有领先性
    缺点：尾部依赖估计需要大样本，cross momentum需要AU数据
    """
    name = "v57_copula_cross_momentum"
    warmup = 400
    freq = "daily"

    tail_period: int = 120
    tail_quantile: float = 0.05
    cross_mom_period: int = 20
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._lower_tail = None
        self._upper_tail = None
        self._cross_mom = None
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

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        n = len(closes)

        # Compute returns for AG
        ag_returns = np.full(n, np.nan)
        ag_returns[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-9)

        # Load AU data
        au_closes = context.load_auxiliary_close("AU")
        if au_closes is not None and len(au_closes) == n:
            au_returns = np.full(n, np.nan)
            au_returns[1:] = np.diff(au_closes) / np.maximum(au_closes[:-1], 1e-9)

            self._lower_tail, self._upper_tail = tail_dependence(
                ag_returns, au_returns, period=self.tail_period, quantile=self.tail_quantile)
            self._cross_mom = cross_momentum(closes, au_closes, period=self.cross_mom_period)
        else:
            # Fallback without AU data
            self._lower_tail = np.full(n, np.nan)
            self._upper_tail = np.full(n, np.nan)
            self._cross_mom = np.full(n, np.nan)
            # Use self-momentum as fallback
            for idx in range(self.cross_mom_period, n):
                prev = closes[idx - self.cross_mom_period]
                if prev > 0:
                    self._cross_mom[idx] = (closes[idx] / prev - 1) * 100

        # Tail dependence change (rising = increasing co-movement)
        self._upper_tail_rising = np.full(n, np.nan)
        self._lower_tail_rising = np.full(n, np.nan)
        lookback = 10
        for idx in range(lookback, n):
            if not np.isnan(self._upper_tail[idx]) and not np.isnan(self._upper_tail[idx - lookback]):
                self._upper_tail_rising[idx] = self._upper_tail[idx] - self._upper_tail[idx - lookback]
            if not np.isnan(self._lower_tail[idx]) and not np.isnan(self._lower_tail[idx - lookback]):
                self._lower_tail_rising[idx] = self._lower_tail[idx] - self._lower_tail[idx - lookback]

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

        ut_rise = self._upper_tail_rising[i]
        lt_rise = self._lower_tail_rising[i]
        cm_val = self._cross_mom[i]
        atr_val = self._atr[i]
        if np.isnan(cm_val) or np.isnan(atr_val) or atr_val <= 0:
            return
        if np.isnan(ut_rise):
            ut_rise = 0.0
        if np.isnan(lt_rise):
            lt_rise = 0.0

        self.bars_since_last_scale += 1

        # ── 1. 止损 ──
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

        # ── 3. 信号反转退出 ──
        if side == 1 and cm_val < 0:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and cm_val > 0:
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场 ──
        if side == 0:
            if ut_rise > 0 and cm_val > 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
            elif lt_rise > 0 and cm_val < 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0

        # ── 5. 加仓 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and cm_val > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and cm_val < 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.sell(add)
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
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
