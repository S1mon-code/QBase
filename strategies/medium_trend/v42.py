import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.momentum.macd import macd
from indicators.volume.obv import obv
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV42(TimeSeriesStrategy):
    """
    策略简介：MACD金叉 + OBV上升趋势确认的30min量价共振策略。

    使用指标：
    - MACD(12,26,9): 动量方向，histogram>0且递增为多头信号
    - OBV: 量价确认，OBV斜率>0表示资金流入
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - MACD histogram > 0（多头动量）
    - OBV 20期斜率 > 0（资金净流入）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - MACD histogram < 0（动量反转）

    优点：量价共振过滤假突破，信号可靠性高
    缺点：OBV在横盘时信号模糊，可能延迟入场
    """
    name = "medium_trend_v42"
    warmup = 600
    freq = "30min"

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    obv_slope_period: int = 20       # Optuna: 10-40
    atr_stop_mult: float = 3.0      # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._macd_hist = None
        self._obv = None
        self._obv_slope = None
        self._atr = None
        self._avg_volume = None

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

        _, _, self._macd_hist = macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        self._obv = obv(closes, volumes)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

        # OBV slope via linear regression over rolling window
        n = len(closes)
        self._obv_slope = np.full(n, np.nan)
        p = self.obv_slope_period
        for j in range(p, n):
            segment = self._obv[j - p:j]
            if np.any(np.isnan(segment)):
                continue
            x = np.arange(p, dtype=np.float64)
            self._obv_slope[j] = np.polyfit(x, segment, 1)[0]

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        hist = self._macd_hist[i]
        obv_sl = self._obv_slope[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(hist) or np.isnan(obv_sl):
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

        # 2. Tiered profit-taking
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

        # 3. Signal exit: MACD histogram flip
        if side == 1 and hist < 0:
            context.close_long()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and hist > 0 and obv_sl > 0:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # 5. Scale-in
        elif side == 1 and self._should_add(price, atr_val, hist, obv_sl):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, hist, obv_sl):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if hist <= 0 or obv_sl <= 0:
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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
