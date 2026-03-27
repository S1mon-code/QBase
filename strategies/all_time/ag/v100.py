import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.psar import psar
from indicators.spread.cross_momentum import cross_momentum
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class PSARCrossMomentumStrategy(TimeSeriesStrategy):
    """
    策略简介：Parabolic SAR 趋势翻转 + 银/铜跨品种动量确认的日频多空策略。

    使用指标：
    - PSAR(0.02,0.02,0.2): 趋势翻转信号，方向从-1变1=多头翻转
    - Cross Momentum(AG vs CU, 20): 银vs铜相对动量，银跑赢铜=做多银
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - PSAR 方向翻转为 1（从空头到多头）
    - AG vs CU cross momentum > 0（银相对铜有正动量）

    进场条件（做空）：
    - PSAR 方向翻转为 -1（从多头到空头）
    - AG vs CU cross momentum < 0（银相对铜有负动量）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - PSAR 方向再次翻转

    优点：PSAR翻转信号明确+跨品种动量提供额外信息维度
    缺点：依赖CU数据同步，PSAR在震荡市频繁翻转
    """
    name = "v100_psar_cross_momentum"
    warmup = 200
    freq = "daily"

    af_start: float = 0.02
    af_step: float = 0.02
    af_max: float = 0.2
    cm_period: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._psar_val = None
        self._psar_dir = None
        self._cross_mom = None
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

        self._psar_val, self._psar_dir = psar(
            highs, lows, self.af_start, self.af_step, self.af_max)
        self._atr = atr(highs, lows, closes, period=14)

        # Load CU auxiliary close for cross momentum
        cu_closes = context.load_auxiliary_close("CU")
        if cu_closes is not None and len(cu_closes) == len(closes):
            self._cross_mom = cross_momentum(closes, cu_closes, self.cm_period)
        else:
            # Fallback: neutral if no CU data
            self._cross_mom = np.zeros(len(closes))

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

        psar_dir = self._psar_dir[i]
        cm_val = self._cross_mom[i]
        atr_val = self._atr[i]
        if np.isnan(psar_dir) or np.isnan(atr_val):
            return
        if np.isnan(cm_val):
            cm_val = 0.0

        self.bars_since_last_scale += 1

        # PSAR flip detection
        if i < 1:
            return
        prev_dir = self._psar_dir[i - 1]
        if np.isnan(prev_dir):
            return
        psar_flip_up = prev_dir == -1 and psar_dir == 1
        psar_flip_down = prev_dir == 1 and psar_dir == -1

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

        # ── 3. PSAR 翻转退出 ──
        if side == 1 and psar_flip_down:
            context.close_long()
            self._reset_state()
        elif side == -1 and psar_flip_up:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if psar_flip_up and cm_val > 0:
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
            elif psar_flip_down and cm_val < 0:
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
                    and psar_dir == 1 and cm_val > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and psar_dir == -1 and cm_val < 0):
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
