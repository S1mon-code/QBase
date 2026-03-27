import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.ml.gaussian_mixture_regime import gmm_regime
from indicators.trend.hma import hma
from indicators.momentum.cci import cci
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV159(TimeSeriesStrategy):
    """
    策略简介：日线GMM regime + 4h HMA趋势 + 1h CCI入场三周期策略。

    使用指标：
    - GMM Regime(3) [日线]: 高斯混合模型识别bull regime
    - HMA(20) [4h]: HMA上升确认中周期方向
    - CCI(20) [1h]: CCI上穿100入场
    - ATR(14) [1h]: 止损距离

    进场条件（做多）：日线GMM=bull + 4h HMA上升 + 1h CCI上穿100
    出场条件：ATR追踪止损, 分层止盈, regime切换

    优点：GMM对异常分布更鲁棒，三层确认
    缺点：GMM拟合可能收敛到局部最优
    """
    name = "medium_trend_v159"
    freq = "1h"
    warmup = 500

    cci_entry: float = 100.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._cci = None
        self._atr = None
        self._avg_volume = None
        self._gmm_d = None
        self._bull_state = None
        self._hma_4h = None
        self._d_map = None
        self._4h_map = None

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

        self._cci = cci(highs, lows, closes, period=20)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step_4h = 4
        n_4h = n // step_4h
        trim_4h = n_4h * step_4h
        closes_4h = closes[:trim_4h].reshape(n_4h, step_4h)[:, -1]
        self._hma_4h = hma(closes_4h, period=20)
        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step_4h - 1), n_4h - 1)

        # GMM on 4h returns+vol features
        returns_4h = np.zeros_like(closes_4h)
        returns_4h[1:] = np.diff(closes_4h) / closes_4h[:-1]
        vol_4h = np.zeros_like(closes_4h)
        for k in range(20, len(closes_4h)):
            vol_4h[k] = np.std(returns_4h[k-20:k])
        features = np.column_stack([returns_4h, vol_4h])
        self._gmm_d = gmm_regime(features, period=120, n_components=3)
        state_means = {}
        for s in range(3):
            mask = self._gmm_d[1:] == s
            state_means[s] = np.nanmean(returns_4h[1:][mask]) if mask.sum() > 0 else -999.0
        self._bull_state = max(state_means, key=state_means.get)
        self._d_map = self._4h_map

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        cci_val = self._cci[i]
        atr_val = self._atr[i]
        regime = self._gmm_d[j]
        h_cur = self._hma_4h[j]
        h_prev = self._hma_4h[j - 1] if j > 0 else np.nan
        if np.isnan(cci_val) or np.isnan(atr_val) or np.isnan(regime) or np.isnan(h_cur) or np.isnan(h_prev):
            return

        prev_cci = self._cci[i - 1] if i > 0 else np.nan
        is_bull = (regime == self._bull_state)
        hma_up = h_cur > h_prev
        cci_cross = (not np.isnan(prev_cci) and prev_cci <= self.cci_entry and cci_val > self.cci_entry)
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

        if side == 1 and not is_bull:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and is_bull and hma_up and cci_cross:
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
                    and is_bull and hma_up):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

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
