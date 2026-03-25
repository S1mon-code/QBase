import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.copula_tail import tail_dependence
from indicators.regime.trend_persistence import trend_persistence
from indicators.volume.cmf import cmf

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV132(TimeSeriesStrategy):
    """
    策略简介：Copula尾部依赖信号 + 趋势持续性质量过滤 + CMF资金流的多空策略。

    使用指标：
    - Copula Tail Dependence(120, 0.05): 检测价格与成交量的尾部共依赖，上尾高=强涨，下尾高=强跌
    - Trend Persistence(20, 60): 自相关趋势持续性，高值=趋势可靠
    - CMF(20): Chaikin资金流，>0买入压力，<0卖出压力

    进场条件（做多）：上尾依赖>下尾 + 趋势持续性>中位数 + CMF>0
    进场条件（做空）：下尾依赖>上尾 + 趋势持续性>中位数 + CMF<0

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - CMF反向或趋势持续性下降

    优点：Copula捕捉极端共同运动，趋势质量过滤减少震荡市交易
    缺点：尾部依赖估计需要大窗口，信号更新缓慢
    """
    name = "ag_alltime_v132"
    warmup = 500
    freq = "daily"

    copula_period: int = 120       # Optuna: 60-200
    copula_quantile: float = 0.05  # Optuna: 0.03-0.10
    persist_period: int = 60       # Optuna: 40-100
    cmf_period: int = 20           # Optuna: 10-30
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._upper_tail = None
        self._lower_tail = None
        self._persistence = None
        self._cmf = None
        self._atr = None
        self._avg_volume = None
        self._persist_median = np.nan

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

        self._atr = atr(highs, lows, closes, period=14)

        # Copula tail dependence between price returns and volume returns
        n = len(closes)
        price_ret = np.full(n, np.nan)
        price_ret[1:] = closes[1:] / np.maximum(closes[:-1], 1e-9) - 1.0
        vol_ret = np.full(n, np.nan)
        vol_ret[1:] = volumes[1:] / np.maximum(volumes[:-1], 1e-9) - 1.0
        self._lower_tail, self._upper_tail = tail_dependence(
            price_ret, vol_ret, period=self.copula_period, quantile=self.copula_quantile)

        # Trend persistence
        self._persistence, _ = trend_persistence(closes, max_lag=20, period=self.persist_period)
        valid_persist = self._persistence[~np.isnan(self._persistence)]
        if len(valid_persist) > 0:
            self._persist_median = np.median(valid_persist)

        # CMF
        self._cmf = cmf(highs, lows, closes, volumes, period=self.cmf_period)

        window = 20
        self._avg_volume = np.full_like(volumes, np.nan)
        for idx in range(window, len(volumes)):
            self._avg_volume[idx] = np.mean(volumes[idx - window:idx])

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
        ut = self._upper_tail[i]
        lt = self._lower_tail[i]
        persist = self._persistence[i]
        cmf_val = self._cmf[i]
        if np.isnan(ut) or np.isnan(lt) or np.isnan(persist) or np.isnan(cmf_val):
            return

        self.bars_since_last_scale += 1

        high_quality = persist > self._persist_median

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

        # 3. Signal-based exit: CMF reversal
        if side == 1 and cmf_val < -0.05:
            context.close_long()
            self._reset_state()
            return
        if side == -1 and cmf_val > 0.05:
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and high_quality:
            if ut > lt and cmf_val > 0:
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
            elif lt > ut and cmf_val < 0:
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
            if self.direction == 1 and cmf_val > 0 and high_quality:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and cmf_val < 0 and high_quality:
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
