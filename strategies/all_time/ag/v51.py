import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.autoencoder_error import reconstruction_error
from indicators.microstructure.amihud import amihud_illiquidity
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class AutoencoderAmihudStrategy(TimeSeriesStrategy):
    """
    策略简介：Autoencoder异常检测 + Amihud流动性分层的多空策略。

    使用指标：
    - Reconstruction Error (Autoencoder proxy): PCA重建误差，高=异常市场状态
    - Amihud Illiquidity: 流动性度量，高=不流动
    - RSI(14): 方向判断辅助
    - ATR(14): 止损距离计算

    策略逻辑：
    - 流动（Amihud zscore < 0）+ 非异常（低重建误差）: 跟随趋势
    - 不流动（Amihud zscore > 1）+ 异常（高重建误差）: 反转交易（fade）
    - 方向由RSI决定：RSI > 50跟随多，RSI < 50跟随空

    出场条件：
    - ATR追踪止损 / 分层止盈 / 信号反转

    优点：自适应流动性环境，异常检测提供额外信息
    缺点：重建误差阈值难以设定，Amihud在极端事件中可能失效
    """
    name = "v51_autoencoder_amihud"
    warmup = 500
    freq = "4h"

    ae_period: int = 120
    amihud_period: int = 20
    anomaly_threshold: float = 2.0  # z-score of reconstruction error
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._recon_error = None
        self._recon_zscore = None
        self._amihud_zscore = None
        self._rsi = None
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

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        n = len(closes)

        rsi_arr = rsi(closes, 14)
        atr_arr = atr(highs, lows, closes, 14)
        returns = np.full(n, np.nan)
        returns[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
        hl_range = (highs - lows) / np.maximum(closes, 1e-9)
        features = np.column_stack([rsi_arr, atr_arr / np.maximum(closes, 1e-9), returns, hl_range])

        self._recon_error = reconstruction_error(features, period=self.ae_period)

        # Z-score of reconstruction error
        self._recon_zscore = np.full(n, np.nan)
        window = 60
        for idx in range(window, n):
            w = self._recon_error[idx - window:idx]
            valid = w[~np.isnan(w)]
            if len(valid) > 10:
                mu = np.mean(valid)
                std = np.std(valid)
                if std > 0 and not np.isnan(self._recon_error[idx]):
                    self._recon_zscore[idx] = (self._recon_error[idx] - mu) / std

        _, self._amihud_zscore = amihud_illiquidity(closes, volumes, period=self.amihud_period)

        self._rsi = rsi(closes, 14)
        self._atr = atr(highs, lows, closes, period=self.atr_period)

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

        recon_z = self._recon_zscore[i]
        amihud_z = self._amihud_zscore[i]
        rsi_val = self._rsi[i]
        atr_val = self._atr[i]
        if np.isnan(recon_z) or np.isnan(amihud_z) or np.isnan(rsi_val) or np.isnan(atr_val) or atr_val <= 0:
            return

        self.bars_since_last_scale += 1

        is_liquid = amihud_z < 0
        is_illiquid = amihud_z > 1.0
        is_anomaly = recon_z > self.anomaly_threshold
        is_normal = recon_z < self.anomaly_threshold

        # Determine direction
        # Liquid + normal: follow trend (RSI direction)
        # Illiquid + anomaly: fade (reverse RSI direction)
        go_long = False
        go_short = False
        if is_liquid and is_normal:
            go_long = rsi_val > 50
            go_short = rsi_val < 50
        elif is_illiquid and is_anomaly:
            go_long = rsi_val < 30  # fade: oversold in illiquid = buy
            go_short = rsi_val > 70  # fade: overbought in illiquid = sell

        # ── 1. 止损 ──
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

        # ── 3. 信号反转退出 ──
        if side == 1 and rsi_val < 40 and is_liquid and is_normal:
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and rsi_val > 60 and is_liquid and is_normal:
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场 ──
        if side == 0:
            if go_long:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
            elif go_short:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0

        # ── 5. 加仓 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and go_long):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and go_short):
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
