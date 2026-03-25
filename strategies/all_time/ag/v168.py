import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.autoencoder_error import reconstruction_error
from indicators.momentum.roc import rate_of_change
from indicators.structure.position_crowding import position_crowding
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class AutoencoderROCCrowdingStrategy(TimeSeriesStrategy):
    """
    策略简介：Autoencoder降噪ROC + Position Crowding极端过滤的4h多空策略。

    使用指标：
    - Reconstruction Error(120,2): 高重建误差=异常市场状态，避免交易
    - ROC(20): 动量方向，降噪后的ROC信号
    - Position Crowding(60): 仓位拥挤度，极端拥挤时减仓/不开仓
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - ROC > 0（正动量）
    - Reconstruction error < 中位数的2倍（市场状态正常）
    - Position crowding不极端（<0.8）

    进场条件（做空）：
    - ROC < 0（负动量）
    - Reconstruction error < 中位数的2倍（市场状态正常）
    - Position crowding不极端（<0.8）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - ROC方向反转 或 crowding极端

    优点：降噪ROC减少假信号+crowding过滤避免拥挤交易被反转
    缺点：autoencoder重建误差阈值难以自适应调整
    """
    name = "v168_autoencoder_roc_crowding"
    warmup = 500
    freq = "4h"

    roc_period: int = 20
    crowding_period: int = 60
    crowding_thresh: float = 0.8
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._roc = None
        self._recon_err = None
        self._recon_median = None
        self._crowding = None
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
        oi = context.get_full_oi_array()

        self._atr = atr(highs, lows, closes, period=14)
        self._roc = rate_of_change(closes, self.roc_period)

        # Autoencoder features
        rets = np.zeros_like(closes)
        rets[1:] = closes[1:] / closes[:-1] - 1
        vol20 = np.full_like(closes, np.nan)
        for idx in range(20, len(closes)):
            vol20[idx] = np.std(rets[idx-20:idx])
        features = np.column_stack([rets, vol20])
        features = np.nan_to_num(features, nan=0.0)
        self._recon_err = reconstruction_error(features, period=120, encoding_dim=2)

        # Rolling median of reconstruction error
        n = len(closes)
        self._recon_median = np.full(n, np.nan)
        window = 120
        for idx in range(window, n):
            vals = self._recon_err[idx-window:idx]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                self._recon_median[idx] = np.median(vals)

        # Position crowding
        self._crowding, _ = position_crowding(closes, oi, volumes, period=self.crowding_period)

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

        roc_val = self._roc[i]
        recon = self._recon_err[i]
        recon_med = self._recon_median[i]
        crowd = self._crowding[i]
        atr_val = self._atr[i]
        if np.isnan(roc_val) or np.isnan(atr_val):
            return
        if np.isnan(recon) or np.isnan(recon_med):
            normal_market = True
        else:
            normal_market = recon < recon_med * 2.0
        if np.isnan(crowd):
            crowd = 0.0

        self.bars_since_last_scale += 1
        not_crowded = abs(crowd) < self.crowding_thresh

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

        # ── 3. ROC反转或crowding极端退出 ──
        if side == 1 and (roc_val < 0 or abs(crowd) > 0.9):
            context.close_long()
            self._reset_state()
        elif side == -1 and (roc_val > 0 or abs(crowd) > 0.9):
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0 and normal_market and not_crowded:
            if roc_val > 0:
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
            elif roc_val < 0:
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
                    and roc_val > 0 and not_crowded):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and roc_val < 0 and not_crowded):
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
