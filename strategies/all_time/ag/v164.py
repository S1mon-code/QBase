import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.gradient_boost_signal import gradient_boost_signal
from indicators.volatility.ttm_squeeze import ttm_squeeze
from indicators.volume.obv import obv
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class RandomForestTTMOBVStrategy(TimeSeriesStrategy):
    """
    策略简介：Random Forest代理（GBT信号）+ TTM Squeeze释放 + OBV确认的4h多空策略。

    使用指标：
    - Gradient Boost Signal(120,20): 替代RF的集成ML方向预测
    - TTM Squeeze(20,2.0,20,1.5): squeeze_on=压缩中，释放时momentum方向入场
    - OBV: 量价趋势确认，OBV上升=资金流入
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - GBT预测 > 0（ML看多）
    - TTM squeeze刚释放（前一bar squeeze=True, 当前=False）且momentum > 0
    - OBV上升（当前OBV > 20bar前OBV）

    进场条件（做空）：
    - GBT预测 < 0（ML看空）
    - TTM squeeze刚释放且momentum < 0
    - OBV下降（当前OBV < 20bar前OBV）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - TTM momentum方向反转

    优点：squeeze释放捕捉波动率扩张+ML过滤假突破+OBV量确认
    缺点：squeeze释放信号稀疏，ML训练窗口可能不够
    """
    name = "v164_rf_ttm_obv"
    warmup = 500
    freq = "4h"

    bb_period: int = 20
    bb_mult: float = 2.0
    kc_period: int = 20
    kc_mult: float = 1.5
    obv_lookback: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._gbt_pred = None
        self._squeeze = None
        self._momentum = None
        self._obv = None
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
        self._squeeze, self._momentum = ttm_squeeze(
            highs, lows, closes, self.bb_period, self.bb_mult, self.kc_period, self.kc_mult)
        self._obv = obv(closes, volumes)

        # ML features
        rets = np.zeros_like(closes)
        rets[1:] = closes[1:] / closes[:-1] - 1
        vol20 = np.full_like(closes, np.nan)
        for idx in range(20, len(closes)):
            vol20[idx] = np.std(rets[idx-20:idx])
        features = np.column_stack([rets, vol20])
        features = np.nan_to_num(features, nan=0.0)
        self._gbt_pred, _ = gradient_boost_signal(closes, features, period=120, n_estimators=20)

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

        if i < self.obv_lookback + 1:
            return

        gbt_val = self._gbt_pred[i]
        sq_now = self._squeeze[i]
        sq_prev = self._squeeze[i - 1]
        mom_val = self._momentum[i]
        obv_now = self._obv[i]
        obv_prev = self._obv[i - self.obv_lookback]
        atr_val = self._atr[i]
        if np.isnan(gbt_val) or np.isnan(mom_val) or np.isnan(atr_val):
            return
        if np.isnan(sq_now) or np.isnan(sq_prev):
            return
        if np.isnan(obv_now) or np.isnan(obv_prev):
            return

        self.bars_since_last_scale += 1
        squeeze_fire = sq_prev > 0.5 and sq_now < 0.5  # squeeze just released
        obv_up = obv_now > obv_prev
        obv_down = obv_now < obv_prev

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

        # ── 3. Momentum反转退出 ──
        if side == 1 and mom_val < 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and mom_val > 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if gbt_val > 0 and squeeze_fire and mom_val > 0 and obv_up:
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
            elif gbt_val < 0 and squeeze_fire and mom_val < 0 and obv_down:
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
                    and mom_val > 0 and obv_up):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and mom_val < 0 and obv_down):
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
