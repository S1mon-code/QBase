import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.boosting_signal import gradient_boost_signal
from indicators.regime.trend_strength_composite import trend_strength
from indicators.volume.obv import obv
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV118(TimeSeriesStrategy):
    """
    策略简介：Gradient Boosting方向预测 + 趋势强度过滤 + OBV量价确认。

    使用指标：
    - Gradient Boosting Signal(120, 20): 机器学习方向预测概率
    - Trend Strength Composite(20): 综合趋势强度评分(0-100)
    - OBV: On-Balance Volume，确认量价配合
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Boosting预测概率>0.6（看涨）
    - 趋势强度>50（足够强的趋势环境）
    - OBV上升（量价配合）
    进场条件（做空）：
    - Boosting预测概率<0.4（看跌）
    - 趋势强度>50
    - OBV下降

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Boosting信号反转或趋势强度下降

    优点：ML模型捕捉非线性关系，多重过滤确保信号质量
    缺点：Boosting存在过拟合风险，需要足够训练数据
    """
    name = "ag_alltime_v118"
    warmup = 500
    freq = "4h"

    boost_period: int = 120       # Optuna: 80-200
    boost_n_est: int = 20         # Optuna: 10-40
    ts_period: int = 20           # Optuna: 10-30
    ts_thresh: float = 50.0       # Optuna: 30-70
    boost_long: float = 0.6       # Optuna: 0.55-0.70
    boost_short: float = 0.4      # Optuna: 0.30-0.45
    obv_slope_period: int = 20    # Optuna: 10-30
    atr_stop_mult: float = 3.0    # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._boost_sig = None
        self._ts = None
        self._obv = None
        self._obv_slope = None
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
        n = len(closes)

        self._atr = atr(highs, lows, closes, period=14)

        # Build features for boosting
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx(highs, lows, closes, period=14)
        atr_arr = self._atr.copy()
        atr_norm = np.full(n, np.nan)
        valid = closes > 1e-9
        atr_norm[valid] = atr_arr[valid] / closes[valid]

        # ROC feature
        roc_arr = np.full(n, np.nan)
        for idx in range(10, n):
            if closes[idx - 10] > 1e-9:
                roc_arr[idx] = (closes[idx] / closes[idx - 10]) - 1.0

        features = np.column_stack([rsi_arr, adx_arr, atr_norm, roc_arr])
        self._boost_sig, _ = gradient_boost_signal(
            closes, features,
            period=self.boost_period, n_estimators=self.boost_n_est,
        )

        # Trend strength
        self._ts = trend_strength(closes, highs, lows, period=self.ts_period)

        # OBV + slope
        self._obv = obv(closes, volumes)
        self._obv_slope = np.full(n, np.nan)
        sp = self.obv_slope_period
        for idx in range(sp, n):
            self._obv_slope[idx] = self._obv[idx] - self._obv[idx - sp]

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
        bsig = self._boost_sig[i]
        ts_val = self._ts[i]
        obv_s = self._obv_slope[i]
        if np.isnan(bsig) or np.isnan(ts_val) or np.isnan(obv_s):
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

        # 3. Signal-based exit: boosting reversal or trend weakening
        if side == 1 and (bsig < 0.45 or ts_val < 30):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (bsig > 0.55 or ts_val < 30):
            context.close_short()
            self._reset_state()
            return

        # Re-read position
        side, lots = context.position

        # 4. Entry: boosting signal + trend strength + OBV
        if side == 0 and ts_val > self.ts_thresh:
            if bsig > self.boost_long and obv_s > 0:
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
            elif bsig < self.boost_short and obv_s < 0:
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
            signal_ok = (self.direction == 1 and bsig > 0.55 and obv_s > 0) or \
                        (self.direction == -1 and bsig < 0.45 and obv_s < 0)
            if signal_ok:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    if self.direction == 1:
                        context.buy(add_lots)
                    else:
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
