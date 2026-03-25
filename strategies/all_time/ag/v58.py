import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.ridge_forecast import rolling_ridge
from indicators.structure.position_crowding import position_crowding
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class ReinforcementCrowdingStrategy(TimeSeriesStrategy):
    """
    策略简介：Ridge回归方向预测（RL proxy）+ 持仓拥挤度调节的多空策略。

    使用指标：
    - Rolling Ridge (RL proxy): 滚动Ridge回归预测方向，作为简化的RL信号
    - Position Crowding: 持仓拥挤度(0-1)，高拥挤=反转风险
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Ridge prediction > 0（预测上涨）
    - Crowding score < 0.7（持仓不拥挤=安全入场）

    进场条件（做空）：
    - Ridge prediction < 0（预测下跌）
    - Crowding score < 0.7

    仓位调节：
    - 当crowding > 0.8时，减半仓位（极端拥挤=反转风险高）

    出场条件：
    - ATR追踪止损 / 分层止盈 / Ridge预测反转

    优点：Ridge在线更新适应市场，拥挤度是期货独有的反转领先指标
    缺点：Ridge线性假设可能不足，拥挤度依赖OI数据质量
    """
    name = "v58_rl_crowding"
    warmup = 400
    freq = "daily"

    ridge_period: int = 120
    crowding_period: int = 60
    crowding_threshold: float = 0.7
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._ridge_pred = None
        self._crowding = None
        self._unwind_risk = None
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
        oi = context.get_full_oi_array()
        n = len(closes)

        rsi_arr = rsi(closes, 14)
        atr_arr = atr(highs, lows, closes, 14)
        returns = np.full(n, np.nan)
        returns[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
        features = np.column_stack([rsi_arr, atr_arr / np.maximum(closes, 1e-9), returns])

        self._ridge_pred, _ = rolling_ridge(
            closes, features, period=self.ridge_period, forecast_horizon=5)

        self._crowding, self._unwind_risk = position_crowding(
            closes, oi, volumes, period=self.crowding_period)

        self._atr = atr(highs, lows, closes, period=self.atr_period)

        window = 20
        self._avg_volume = np.full_like(volumes, np.nan)
        for idx in range(window, n):
            self._avg_volume[idx] = np.mean(volumes[idx - window:idx])

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        ridge_val = self._ridge_pred[i]
        crowd_val = self._crowding[i]
        atr_val = self._atr[i]
        if np.isnan(ridge_val) or np.isnan(atr_val) or atr_val <= 0:
            return
        if np.isnan(crowd_val):
            crowd_val = 0.0

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

        # ── Extreme crowding: reduce position ──
        if crowd_val > 0.8 and (side == 1 or side == -1) and lots > 1:
            reduce = max(1, lots // 2)
            if side == 1:
                context.close_long(lots=reduce)
            else:
                context.close_short(lots=reduce)
            return

        # ── 3. 信号反转退出 ──
        if side == 1 and ridge_val < 0:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and ridge_val > 0:
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场 ──
        if side == 0 and crowd_val < self.crowding_threshold:
            if ridge_val > 0:
                base_lots = self._calc_lots(context, atr_val)
                # Reduce lots if crowding moderate
                if crowd_val > 0.5:
                    base_lots = max(1, base_lots // 2)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
            elif ridge_val < 0:
                base_lots = self._calc_lots(context, atr_val)
                if crowd_val > 0.5:
                    base_lots = max(1, base_lots // 2)
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
                    and ridge_val > 0 and crowd_val < 0.5):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and ridge_val < 0 and crowd_val < 0.5):
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
