import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.psar import psar
from indicators.volatility.hurst import hurst_exponent
from indicators.structure.smart_money import smart_money_index
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV193(TimeSeriesStrategy):
    """
    策略简介：PSAR翻转信号 + Hurst趋势性确认 + Smart Money方向的趋势策略。

    使用指标：
    - PSAR(0.02, 0.02, 0.2): 趋势翻转信号，SAR点位切换
    - Hurst Exponent(20): H>0.5确认趋势市场，H<0.5回避震荡
    - Smart Money Index(20): 机构资金方向，SMI上升=聪明钱做多
    - ATR(14): 止损距离计算

    进场条件（做多）：PSAR由上翻下（看涨）+ Hurst>0.55（趋势市）+ SMI上升
    进场条件（做空）：PSAR由下翻上（看跌）+ Hurst>0.55（趋势市）+ SMI下降

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - PSAR反向翻转

    优点：Hurst过滤震荡市减少PSAR假信号，Smart Money提供机构确认
    缺点：Hurst计算可能滞后，Smart Money在日内数据上近似精度有限
    """
    name = "ag_alltime_v193"
    warmup = 200
    freq = "4h"

    psar_af_start: float = 0.02   # Optuna: 0.01-0.04
    psar_af_step: float = 0.02    # Optuna: 0.01-0.04
    psar_af_max: float = 0.2      # Optuna: 0.1-0.3
    hurst_lag: int = 20           # Optuna: 15-30
    hurst_thresh: float = 0.55    # Optuna: 0.50-0.65
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._psar_val = None
        self._psar_dir = None
        self._hurst = None
        self._smi = None
        self._smi_sig = None
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

        self._psar_val, self._psar_dir = psar(
            highs, lows, self.psar_af_start, self.psar_af_step, self.psar_af_max)
        self._hurst = hurst_exponent(closes, max_lag=self.hurst_lag)
        self._smi, self._smi_sig = smart_money_index(
            opens, closes, highs, lows, volumes, period=20)
        self._atr = atr(highs, lows, closes, period=14)

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
        pdir = self._psar_dir[i]
        h_val = self._hurst[i]
        smi_val = self._smi[i]

        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(pdir):
            return
        if np.isnan(h_val):
            h_val = 0.5
        if np.isnan(smi_val):
            return

        prev_dir = self._psar_dir[i - 1] if i > 0 else np.nan
        prev_smi = self._smi[i - 1] if i > 0 else np.nan
        if np.isnan(prev_dir) or np.isnan(prev_smi):
            return

        smi_rising = smi_val > prev_smi
        smi_falling = smi_val < prev_smi

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

        # 3. Signal exit: PSAR flip
        if side == 1 and pdir == -1 and prev_dir == 1:
            context.close_long()
            self._reset_state()
        elif side == -1 and pdir == 1 and prev_dir == -1:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: PSAR flip + trending Hurst + smart money direction
        if side == 0:
            if prev_dir == -1 and pdir == 1 and h_val > self.hurst_thresh and smi_rising:
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
            elif prev_dir == 1 and pdir == -1 and h_val > self.hurst_thresh and smi_falling:
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
            if self.direction == 1 and pdir == 1 and smi_rising:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and pdir == -1 and smi_falling:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
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
