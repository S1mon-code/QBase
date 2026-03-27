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
from indicators.trend.ema import ema
from indicators.volatility.bollinger import bollinger_bands
from indicators.volume.klinger import klinger

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


class StrategyV194(TimeSeriesStrategy):
    """
    策略简介：Dual-mode: HMM regime(1,2)→trend(EMA 20/50 cross), HMM(0)→revert(Bollinger) + Klinger.

    使用指标：
    - HMM Regime(120): trend(1,2) vs ranging(0) detection
    - EMA(20) vs EMA(50): trend-following crossover in trend mode
    - Bollinger(20,2): band touch reversion in range mode
    - Klinger(34,55,13): volume flow confirmation
    - ATR(14): stop distance

    进场条件（趋势-做多）：HMM=1 且 EMA20>EMA50 且 Klinger>signal
    进场条件（趋势-做空）：HMM=2 且 EMA20<EMA50 且 Klinger<signal
    进场条件（回归-做多）：HMM=0 且 price<BB lower 且 Klinger>signal
    进场条件（回归-做空）：HMM=0 且 price>BB upper 且 Klinger<signal

    出场条件：ATR trailing stop / tiered profit / mode-specific exit

    优点：HMM provides probabilistic regime classification
    缺点：HMM regime detection has lag, EMA cross also lags
    """
    name = "i_alltime_v194"
    warmup = 400
    freq = "4h"

    ema_fast: int = 20
    ema_slow: int = 50
    bb_period: int = 20
    regime_period: int = 120
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._regime = None
        self._ema_f = None
        self._ema_s = None
        self._bb_upper = None
        self._bb_lower = None
        self._bb_mid = None
        self._klinger_line = None
        self._klinger_sig = None
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
        self._ema_f = ema(closes, self.ema_fast)
        self._ema_s = ema(closes, self.ema_slow)
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(closes, self.bb_period, 2.0)
        self._klinger_line, self._klinger_sig = klinger(highs, lows, closes, volumes)
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
        ef = self._ema_f[i]
        es = self._ema_s[i]
        bb_u = self._bb_upper[i]
        bb_l = self._bb_lower[i]
        bb_m = self._bb_mid[i]
        kl = self._klinger_line[i]
        ks = self._klinger_sig[i]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(ef) or np.isnan(es) or np.isnan(bb_u) or np.isnan(kl) or np.isnan(ks):
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
            if is_trend and ef < es:
                context.close_long()
                self._reset_state()
                return
            elif is_range and price > bb_m:
                context.close_long()
                self._reset_state()
                return
        elif side == -1:
            if is_trend and ef > es:
                context.close_short()
                self._reset_state()
                return
            elif is_range and price < bb_m:
                context.close_short()
                self._reset_state()
                return

        # 4. Entry
        if side == 0:
            if (regime == 1 and ef > es and kl > ks) or (is_range and price < bb_l and kl > ks):
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
            elif (regime == 2 and ef < es and kl < ks) or (is_range and price > bb_u and kl < ks):
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
