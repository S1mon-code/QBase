import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.regime.momentum_regime import momentum_regime
from indicators.volatility.ttm_squeeze import ttm_squeeze
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV137(TimeSeriesStrategy):
    """
    策略简介：4h动量regime过滤 + 10min TTM Squeeze突破入场。

    使用指标：
    - Momentum Regime(10, 60) [4h]: 识别正动量regime
    - TTM Squeeze(20, 2.0, 20, 1.5) [10min]: 波动压缩后突破
    - ATR(14) [10min]: 止损距离

    进场条件（做多）：4h regime=正动量, 10min squeeze释放且momentum>0
    出场条件：ATR追踪止损, 分层止盈, regime切换

    优点：TTM Squeeze捕捉压缩后爆发，regime过滤方向
    缺点：Squeeze释放不一定往上
    """
    name = "medium_trend_v137"
    freq = "10min"
    warmup = 2000

    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._squeeze = None
        self._squeeze_mom = None
        self._atr = None
        self._avg_volume = None
        self._regime_4h = None
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

        self._squeeze, self._squeeze_mom = ttm_squeeze(highs, lows, closes)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        step = 24  # 10min * 24 = 4h
        n_4h = n // step
        trim = n_4h * step
        closes_4h = closes[:trim].reshape(n_4h, step)[:, -1]

        self._regime_4h = momentum_regime(closes_4h, fast=10, slow=60)
        self._4h_map = np.minimum(np.maximum(0, (np.arange(n) + 1) // step - 1),
                                  len(self._regime_4h) - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        sq = self._squeeze[i]
        sq_mom = self._squeeze_mom[i]
        atr_val = self._atr[i]
        regime = self._regime_4h[j]
        if np.isnan(sq) or np.isnan(sq_mom) or np.isnan(atr_val) or np.isnan(regime):
            return

        prev_sq = self._squeeze[i - 1] if i > 0 else np.nan
        bull_regime = (regime == 1)
        squeeze_fire = (not np.isnan(prev_sq) and prev_sq == 1 and sq == 0 and sq_mom > 0)
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

        if side == 1 and regime == -1:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and bull_regime and squeeze_fire:
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
                    and bull_regime and sq_mom > 0):
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
