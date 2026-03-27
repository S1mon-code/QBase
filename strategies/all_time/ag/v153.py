import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.elastic_net_forecast import elastic_net_signal
from indicators.regime.trend_strength_composite import trend_strength
from indicators.microstructure.price_efficiency import price_efficiency_coefficient

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV153(TimeSeriesStrategy):
    """
    策略简介：Elastic Net预测 + 趋势质量(高质量趋势) + 价格效率(低效率=机会)的多空策略。

    使用指标：
    - Elastic Net(120): L1+L2正则化回归预测方向
    - Trend Strength(20): 趋势质量复合评分(0-100)，高值=强趋势
    - Price Efficiency(20): 价格效率系数，低值=价格无效（有获利空间）
    - ATR(14): 止损距离计算

    进场条件（做多）：Elastic Net信号>0 + 趋势质量>50 + PEC<0.5(低效率)
    进场条件（做空）：Elastic Net信号<0 + 趋势质量>50 + PEC<0.5

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Elastic Net信号反转

    优点：Elastic Net结合L1特征选择+L2稳定性，趋势质量多维确认
    缺点：日频信号稀疏，滚动训练计算量大
    """
    name = "ag_alltime_v153"
    warmup = 250
    freq = "daily"

    ts_threshold: float = 50.0     # Optuna: 30-70
    pec_threshold: float = 0.5     # Optuna: 0.3-0.7
    atr_stop_mult: float = 3.5    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._avg_volume = None
        self._enet_signal = None
        self._enet_conf = None
        self._ts = None
        self._pec = None

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
        self._ts = trend_strength(closes, highs, lows, period=20)
        self._pec, _ = price_efficiency_coefficient(closes, period=20)

        # Build features for Elastic Net
        from indicators.momentum.rsi import rsi
        from indicators.trend.adx import adx as adx_fn
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx_fn(highs, lows, closes, period=14)
        atr_arr = self._atr
        features = np.column_stack([rsi_arr, adx_arr, atr_arr])

        self._enet_signal, self._enet_conf = elastic_net_signal(
            closes, features, period=120, alpha=0.1, l1_ratio=0.5
        )

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

        enet_sig = self._enet_signal[i]
        ts_val = self._ts[i]
        pec_val = self._pec[i]
        if np.isnan(enet_sig) or np.isnan(ts_val) or np.isnan(pec_val):
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

        # 3. Signal exit: Elastic Net reversal
        if side == 1 and enet_sig < 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and enet_sig > 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry
        if side == 0 and ts_val > self.ts_threshold and pec_val < self.pec_threshold:
            if enet_sig > 0:
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
            elif enet_sig < 0:
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
        elif side != 0 and self._should_add(price, atr_val, enet_sig):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if self.direction == 1:
                    context.buy(add_lots)
                else:
                    context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, enet_sig):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1:
            if price < self.entry_price + atr_val:
                return False
            if enet_sig <= 0:
                return False
        elif self.direction == -1:
            if price > self.entry_price - atr_val:
                return False
            if enet_sig >= 0:
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
