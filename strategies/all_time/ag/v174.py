import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.ridge_forecast import rolling_ridge
from indicators.trend.aroon import aroon
from indicators.spread.relative_strength import relative_strength
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class RLAroonRelativeStrengthStrategy(TimeSeriesStrategy):
    """
    策略简介：RL代理（Ridge预测）+ Aroon趋势确认 + 银相对强度的日频多空策略。

    使用指标：
    - Rolling Ridge(120,5): 替代RL的ML方向预测
    - Aroon(25): Aroon Up>70=强上涨趋势，Aroon Down>70=强下跌趋势
    - Relative Strength(AG vs AU, 20): 银相对金的强度，正=银强于金
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Ridge forecast > 0（ML看多）
    - Aroon Up > 70 且 Aroon Osc > 0（上涨趋势确认）
    - Relative Strength ratio上升（银强于金）

    进场条件（做空）：
    - Ridge forecast < 0（ML看空）
    - Aroon Down > 70 且 Aroon Osc < 0（下跌趋势确认）
    - Relative Strength ratio下降（银弱于金）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - Aroon Osc方向反转

    优点：Aroon对趋势强度判断清晰+相对强度提供跨品种确认
    缺点：Aroon在区间震荡中频繁变换，相对强度可能与绝对价格脱钩
    """
    name = "v174_rl_aroon_relative_strength"
    warmup = 400
    freq = "daily"

    aroon_period: int = 25
    aroon_thresh: float = 70.0
    rs_period: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._ridge_pred = None
        self._aroon_up = None
        self._aroon_down = None
        self._aroon_osc = None
        self._rs = None
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

        self._atr = atr(highs, lows, closes, period=14)
        self._aroon_up, self._aroon_down, self._aroon_osc = aroon(
            highs, lows, period=self.aroon_period)

        # ML features
        rets = np.zeros_like(closes)
        rets[1:] = closes[1:] / closes[:-1] - 1
        vol20 = np.full_like(closes, np.nan)
        for idx in range(20, len(closes)):
            vol20[idx] = np.std(rets[idx-20:idx])
        features = np.column_stack([rets, vol20])
        features = np.nan_to_num(features, nan=0.0)
        self._ridge_pred, _ = rolling_ridge(closes, features, period=120, forecast_horizon=5)

        # Relative strength AG vs AU
        au_closes = context.load_auxiliary_close("AU")
        if au_closes is not None and len(au_closes) == len(closes):
            self._rs, _, _ = relative_strength(closes, au_closes, period=self.rs_period)
        else:
            self._rs = np.zeros(len(closes))

        window = 20
        self._avg_volume = np.full_like(volumes, np.nan)
        for idx in range(window, len(volumes)):
            self._avg_volume[idx] = np.mean(volumes[idx-window:idx])

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        if i < 2:
            return
        ridge_val = self._ridge_pred[i]
        aroon_u = self._aroon_up[i]
        aroon_d = self._aroon_down[i]
        aroon_o = self._aroon_osc[i]
        rs_now = self._rs[i]
        rs_prev = self._rs[i - 1]
        atr_val = self._atr[i]
        if np.isnan(ridge_val) or np.isnan(aroon_u) or np.isnan(aroon_o) or np.isnan(atr_val):
            return
        if np.isnan(rs_now) or np.isnan(rs_prev):
            rs_now = 0.0
            rs_prev = 0.0

        self.bars_since_last_scale += 1
        rs_rising = rs_now > rs_prev
        rs_falling = rs_now < rs_prev

        # ── 1. 止损检查 ──
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

        # ── 3. Aroon Osc反转退出 ──
        if side == 1 and aroon_o < 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and aroon_o > 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if ridge_val > 0 and aroon_u > self.aroon_thresh and aroon_o > 0 and rs_rising:
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
            elif ridge_val < 0 and aroon_d > self.aroon_thresh and aroon_o < 0 and rs_falling:
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

        # ── 5. 加仓逻辑 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and aroon_o > 0 and rs_rising):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and aroon_o < 0 and rs_falling):
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
        self.direction = 0
