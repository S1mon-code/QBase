import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.regime.vol_regime_markov import vol_regime_simple
from indicators.volatility.ttm_squeeze import ttm_squeeze
from indicators.structure.smart_money import smart_money_index
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class VolRegimeTTMSmartMoneyStrategy(TimeSeriesStrategy):
    """
    策略简介：Vol Regime扩张检测 + TTM Squeeze释放 + Smart Money方向的4h多空策略。

    使用指标：
    - Vol Regime Simple(60): 高/低波动率regime识别
    - TTM Squeeze(20,2.0,20,1.5): squeeze释放=波动率爆发信号
    - Smart Money Index(20): 机构资金方向推断
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Vol Regime = 高波动率（regime=1，expanding vol）
    - TTM squeeze刚释放且momentum > 0
    - Smart Money Index上升（机构看多）

    进场条件（做空）：
    - Vol Regime = 高波动率
    - TTM squeeze刚释放且momentum < 0
    - Smart Money Index下降（机构看空）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - TTM momentum方向反转

    优点：vol regime确保在波动扩张期交易+squeeze提供精确时机+smart money方向
    缺点：squeeze信号稀疏可能等待过久，smart money推断不够精确
    """
    name = "v179_vol_regime_ttm_smart_money"
    warmup = 500
    freq = "4h"

    vol_period: int = 60
    bb_period: int = 20
    bb_mult: float = 2.0
    kc_period: int = 20
    kc_mult: float = 1.5
    sm_period: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._vol_reg = None
        self._squeeze = None
        self._momentum = None
        self._sm_idx = None
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
        opens = context.get_full_open_array()
        volumes = context.get_full_volume_array()

        self._atr = atr(highs, lows, closes, period=14)
        self._vol_reg, _ = vol_regime_simple(closes, period=self.vol_period)
        self._squeeze, self._momentum = ttm_squeeze(
            highs, lows, closes, self.bb_period, self.bb_mult, self.kc_period, self.kc_mult)
        self._sm_idx, _ = smart_money_index(opens, closes, highs, lows, volumes,
                                             period=self.sm_period)

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

        if i < 2:
            return
        vol_reg = self._vol_reg[i]
        sq_now = self._squeeze[i]
        sq_prev = self._squeeze[i - 1]
        mom_val = self._momentum[i]
        sm_now = self._sm_idx[i]
        sm_prev = self._sm_idx[i - 1]
        atr_val = self._atr[i]
        if np.isnan(vol_reg) or np.isnan(mom_val) or np.isnan(atr_val):
            return
        if np.isnan(sq_now) or np.isnan(sq_prev):
            return
        if np.isnan(sm_now) or np.isnan(sm_prev):
            sm_now = 0.0
            sm_prev = 0.0

        self.bars_since_last_scale += 1
        high_vol = vol_reg == 1
        squeeze_fire = sq_prev > 0.5 and sq_now < 0.5
        sm_up = sm_now > sm_prev
        sm_down = sm_now < sm_prev

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
        if side == 0 and high_vol and squeeze_fire:
            if mom_val > 0 and sm_up:
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
            elif mom_val < 0 and sm_down:
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
                    and mom_val > 0 and sm_up):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and mom_val < 0 and sm_down):
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
