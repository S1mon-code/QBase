import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.isolation_anomaly import isolation_anomaly
from indicators.regime.vol_regime_markov import vol_regime_simple
from indicators.volume.klinger import klinger

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV130(TimeSeriesStrategy):
    """
    策略简介：Isolation Forest 异常过滤 + 波动率状态 + Klinger 成交量方向的多空策略。

    使用指标：
    - Isolation Anomaly(120): 非异常bars上交易（异常bars跳过）
    - Vol Regime Markov(60): 波动率扩张状态有利于趋势延续
    - Klinger(34,55,13): 成交量加权趋势方向

    进场条件（做多）：非异常，Vol 扩张态，Klinger KVO > signal
    进场条件（做空）：非异常，Vol 扩张态，Klinger KVO < signal

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - Klinger 反转或检测到异常

    优点：异常过滤避免极端行情下假信号，Klinger 综合量价信息
    缺点：Isolation Forest 边界不稳定，可能错过正常突破
    """
    name = "ag_alltime_v130"
    warmup = 400
    freq = "4h"

    iso_period: int = 120
    iso_contamination: float = 0.05
    vol_regime_period: int = 60
    klinger_fast: int = 34
    klinger_slow: int = 55
    klinger_signal: int = 13
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._anomaly = None
        self._vol_regime = None
        self._klinger_kvo = None
        self._klinger_sig = None
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

        from indicators.momentum.rsi import rsi
        rsi_arr = rsi(closes, period=14)
        features = np.column_stack([rsi_arr, self._atr, closes])
        self._anomaly = isolation_anomaly(features, period=self.iso_period, contamination=self.iso_contamination)

        self._vol_regime = vol_regime_simple(closes, period=self.vol_regime_period)

        kvo, sig = klinger(highs, lows, closes, volumes,
                           fast=self.klinger_fast, slow=self.klinger_slow, signal=self.klinger_signal)
        self._klinger_kvo = kvo
        self._klinger_sig = sig

        window = 20
        self._avg_volume = np.full_like(volumes, np.nan)
        for idx in range(window, len(volumes)):
            self._avg_volume[idx] = np.mean(volumes[idx - window:idx])

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        anom_val = self._anomaly[i]
        vr_val = self._vol_regime[i]
        kvo_val = self._klinger_kvo[i]
        ksig_val = self._klinger_sig[i]
        if np.isnan(anom_val) or np.isnan(vr_val) or np.isnan(kvo_val) or np.isnan(ksig_val):
            return

        # anomaly: -1 = anomaly, 1 = normal
        is_normal = anom_val >= 0
        # vol regime: high values = expanding
        vol_expanding = vr_val > 0

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

        # 3. Signal-based exit: anomaly or Klinger reversal
        if side == 1 and (not is_normal or kvo_val < ksig_val):
            context.close_long()
            self._reset_state()
            return
        if side == -1 and (not is_normal or kvo_val > ksig_val):
            context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and is_normal and vol_expanding:
            if kvo_val > ksig_val:
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
            elif kvo_val < ksig_val:
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
            if self.direction == 1 and kvo_val > ksig_val and is_normal:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and kvo_val < ksig_val and is_normal:
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
