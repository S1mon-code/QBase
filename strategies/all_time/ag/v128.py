import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.autoencoder_error import reconstruction_error
from indicators.regime.sample_entropy import sample_entropy
from indicators.microstructure.amihud import amihud_illiquidity

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV128(TimeSeriesStrategy):
    """
    策略简介：Autoencoder 去噪信号 + 低噪行情过滤 + Amihud 流动性的多空策略。

    使用指标：
    - Reconstruction Error(120): AE 重构误差低=正常行情（高=异常）
    - Sample Entropy(60): 低熵=有序/低噪，高熵=无序/高噪
    - Amihud Illiquidity(20): 正常流动性下信号更可靠

    进场条件（做多）：AE 重构误差低（正常），Sample Entropy低（低噪），
                     Amihud 正常，价格在 Kalman 趋势上方
    进场条件（做空）：同理但价格在趋势下方

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - AE 重构误差突然升高（异常行情）

    优点：双重噪音过滤大幅减少假信号，流动性过滤避免滑点陷阱
    缺点：过滤条件严格导致交易频率低
    """
    name = "ag_alltime_v128"
    warmup = 400
    freq = "4h"

    ae_period: int = 120
    entropy_period: int = 60
    entropy_thresh: float = 0.5
    amihud_period: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._recon_err = None
        self._entropy = None
        self._amihud = None
        self._kalman = None
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

        # Features for autoencoder
        from indicators.momentum.rsi import rsi
        from indicators.trend.adx import adx as adx_fn
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx_fn(highs, lows, closes, period=14)
        features = np.column_stack([rsi_arr, adx_arr, self._atr])
        self._recon_err = reconstruction_error(features, period=self.ae_period, encoding_dim=2)

        self._entropy = sample_entropy(closes, m=2, r_mult=0.2, period=self.entropy_period)
        self._amihud = amihud_illiquidity(closes, volumes, period=self.amihud_period)

        # Kalman for trend direction
        from indicators.ml.kalman_trend import kalman_filter
        self._kalman = kalman_filter(closes)

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
        if np.isnan(atr_val) or atr_val <= 0:
            return
        err_val = self._recon_err[i]
        ent_val = self._entropy[i]
        ami_val = self._amihud[i]
        kal_val = self._kalman[i]
        if np.isnan(err_val) or np.isnan(ent_val) or np.isnan(ami_val) or np.isnan(kal_val):
            return

        # Compute thresholds: low recon error and low entropy
        # Use rolling median for dynamic threshold
        lookback = min(i + 1, 120)
        err_window = self._recon_err[max(0, i - lookback + 1):i + 1]
        err_window = err_window[~np.isnan(err_window)]
        err_normal = len(err_window) > 10 and err_val < np.median(err_window) * 1.5

        low_noise = ent_val < self.entropy_thresh

        # Amihud: normal liquidity (not extreme illiquidity)
        ami_window = self._amihud[max(0, i - lookback + 1):i + 1]
        ami_window = ami_window[~np.isnan(ami_window)]
        ami_normal = len(ami_window) > 10 and ami_val < np.percentile(ami_window, 80)

        # Price vs Kalman trend
        closes_arr = context.get_full_close_array()
        close_val = closes_arr[i]
        trend_up = close_val > kal_val
        trend_down = close_val < kal_val

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

        # 3. Signal-based exit: anomaly detected
        if side != 0 and not err_normal:
            if side == 1:
                context.close_long()
            else:
                context.close_short()
            self._reset_state()
            return

        # 4. Entry
        if side == 0 and err_normal and low_noise and ami_normal:
            if trend_up:
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
            elif trend_down:
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
            if self.direction == 1 and trend_up and err_normal:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and trend_down and err_normal:
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
