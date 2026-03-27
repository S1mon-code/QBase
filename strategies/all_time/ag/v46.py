import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.feature_importance import rolling_tree_importance
from indicators.volatility.ttm_squeeze import ttm_squeeze
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class RandomForestTTMStrategy(TimeSeriesStrategy):
    """
    策略简介：Random Forest特征重要性方向 + TTM Squeeze释放入场。

    使用指标：
    - Random Forest (rolling_tree_importance): 训练RF分类器预测方向，
      用特征重要性加权信号方向
    - TTM Squeeze: BB收入KC通道内=蓄势，释放时动量方向入场
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - TTM squeeze刚释放（前一bar squeeze_on，当前bar squeeze_off）
    - TTM momentum > 0（向上动量）
    - RF信号特征显示看多倾向

    进场条件（做空）：
    - TTM squeeze刚释放
    - TTM momentum < 0（向下动量）
    - RF信号特征显示看空倾向

    出场条件：
    - ATR追踪止损 / 分层止盈 / 动量反转

    优点：TTM Squeeze精确捕捉波动率收缩后的爆发，RF提供方向确认
    缺点：Squeeze释放后可能假突破，RF过拟合风险
    """
    name = "v46_rf_ttm_squeeze"
    warmup = 500
    freq = "4h"

    rf_period: int = 120
    bb_period: int = 20
    kc_mult: float = 1.5
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._squeeze_on = None
        self._ttm_momentum = None
        self._rf_importance = None
        self._rf_signal = None
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

        self._squeeze_on, self._ttm_momentum = ttm_squeeze(
            highs, lows, closes, bb_period=self.bb_period, kc_mult=self.kc_mult)

        rsi_arr = rsi(closes, 14)
        atr_arr = atr(highs, lows, closes, 14)
        returns = np.full(n, np.nan)
        returns[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
        features = np.column_stack([rsi_arr, atr_arr / np.maximum(closes, 1e-9), returns])

        # RF importance matrix (N, K) - use sum of importances as direction proxy
        self._rf_importance = rolling_tree_importance(
            closes, features, period=self.rf_period)
        # Build a direction signal from RF: predict returns sign using importance-weighted features
        self._rf_signal = np.full(n, np.nan)
        for idx in range(self.rf_period + 1, n):
            imp = self._rf_importance[idx]
            if np.any(np.isnan(imp)):
                continue
            feat = features[idx]
            if np.any(np.isnan(feat)):
                continue
            # Importance-weighted feature sum as directional signal
            self._rf_signal[idx] = np.sum(imp * feat)

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

        if i < 1:
            return
        squeeze_now = self._squeeze_on[i]
        squeeze_prev = self._squeeze_on[i - 1]
        mom_val = self._ttm_momentum[i]
        rf_val = self._rf_signal[i]
        atr_val = self._atr[i]
        if np.isnan(mom_val) or np.isnan(rf_val) or np.isnan(atr_val) or atr_val <= 0:
            return

        self.bars_since_last_scale += 1
        squeeze_fire = squeeze_prev and not squeeze_now  # squeeze just released

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
        if side == 1 and mom_val < 0:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and mom_val > 0:
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场 ──
        if side == 0 and squeeze_fire:
            if mom_val > 0 and rf_val > 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
            elif mom_val < 0 and rf_val < 0:
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
                    and mom_val > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and mom_val < 0):
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
