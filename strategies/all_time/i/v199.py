import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.i.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.volatility.atr import atr
from indicators.trend.psar import psar
from indicators.momentum.williams_r import williams_r
from indicators.volume.volume_spike import volume_spike

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


def _hmm_regime(closes, period=120):
    """Simple regime detection via rolling z-score of log returns."""
    n = len(closes)
    states = np.full(n, 0, dtype=np.int32)
    state_prob = np.full(n, np.nan)
    if n < period + 1:
        return states, state_prob
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-9))
    for i in range(period, n):
        window = log_ret[i - period + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        mu, sigma = np.mean(valid), np.std(valid)
        if sigma < 1e-12:
            states[i] = 0
            state_prob[i] = 0.5
            continue
        z = mu / sigma * np.sqrt(len(valid))
        if z > 1.0:
            states[i] = 1
            state_prob[i] = min(1.0, abs(z) / 3.0)
        elif z < -1.0:
            states[i] = 2
            state_prob[i] = min(1.0, abs(z) / 3.0)
        else:
            states[i] = 0
            state_prob[i] = 1.0 - abs(z)
    return states, state_prob


class StrategyV199(TimeSeriesStrategy):
    """
    策略简介：Dual-mode: HMM(1,2)→trend(PSAR direction), HMM(0)→revert(Williams %R) + Volume Spike on 1h.

    使用指标：
    - HMM Regime(120): market state detection
    - PSAR(0.02,0.02,0.2): trend direction in trend mode
    - Williams %R(14): overbought/oversold in range mode
    - Volume Spike(20,1.5): volume confirmation
    - ATR(14): stop distance

    进场条件（趋势-做多）：HMM=1 且 PSAR direction=1 且 Volume Spike
    进场条件（趋势-做空）：HMM=2 且 PSAR direction=-1 且 Volume Spike
    进场条件（回归-做多）：HMM=0 且 Williams %R < -80 且 Volume Spike
    进场条件（回归-做空）：HMM=0 且 Williams %R > -20 且 Volume Spike

    出场条件：ATR trailing stop / tiered profit / mode exit

    优点：PSAR provides clear trailing stop levels in trend mode
    缺点：PSAR whipsaws in sideways markets (handled by mode switch)
    """
    name = "i_alltime_v199"
    warmup = 600
    freq = "1h"

    wr_period: int = 14
    wr_oversold: float = -80.0
    wr_overbought: float = -20.0
    regime_period: int = 120
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._regime = None
        self._psar_val = None
        self._psar_dir = None
        self._wr = None
        self._vol_spike = None
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
        self._regime, _ = _hmm_regime(closes, self.regime_period)
        self._psar_val, self._psar_dir = psar(highs, lows)
        self._wr = williams_r(highs, lows, closes, self.wr_period)
        self._vol_spike = volume_spike(volumes, 20, 1.5)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        regime = self._regime[i]
        pd = self._psar_dir[i]
        wr = self._wr[i]
        vs = self._vol_spike[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(pd) or np.isnan(wr):
            return

        is_trend = regime in (1, 2)
        is_range = regime == 0

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
            profit_atr = ((price - self.entry_price) / atr_val) if side == 1 else ((self.entry_price - price) / atr_val)
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                cl = max(1, lots // 3)
                (context.close_long if side == 1 else context.close_short)(lots=cl)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                cl = max(1, lots // 3)
                (context.close_long if side == 1 else context.close_short)(lots=cl)
                self._took_profit_3atr = True
                return

        # 3. Signal exit
        if side == 1:
            if is_trend and pd == -1:
                context.close_long()
                self._reset_state()
                return
            elif is_range and wr > -50:
                context.close_long()
                self._reset_state()
                return
        elif side == -1:
            if is_trend and pd == 1:
                context.close_short()
                self._reset_state()
                return
            elif is_range and wr < -50:
                context.close_short()
                self._reset_state()
                return

        # 4. Entry
        if side == 0:
            if (regime == 1 and pd == 1 and vs) or (is_range and wr < self.wr_oversold and vs):
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
            elif (regime == 2 and pd == -1 and vs) or (is_range and wr > self.wr_overbought and vs):
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
        if side != 0 and self._should_add(price, atr_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if side == 1:
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
        spec = _SPEC_MANAGER.get(context.symbol)
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
