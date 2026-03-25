import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.ensemble_signal import ensemble_vote
from indicators.regime.fractal_dimension import fractal_dim
from indicators.trend.psar import psar
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi
from indicators.trend.adx import adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class EnsembleMLFractalDimPSARStrategy(TimeSeriesStrategy):
    """
    策略简介：PSAR 翻转信号 + ML Ensemble 方向确认 + 低分形维度过滤的 4h 多空策略。

    使用指标：
    - PSAR(0.02, 0.02, 0.2): 趋势翻转信号
    - Ensemble Vote(Ridge+Lasso+RF, 120): ML 集成方向预测
    - Fractal Dimension(60): ~1.0=光滑趋势(好), ~1.5=随机游走(坏), ~2.0=噪音(坏)
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - PSAR 翻转为多头（方向从-1变1）
    - Ensemble vote > 0（ML 看多）
    - Fractal dim < 1.4（市场处于有序趋势状态）

    进场条件（做空）：
    - PSAR 翻转为空头
    - Ensemble vote < 0
    - Fractal dim < 1.4

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - PSAR 再次翻转

    优点：PSAR 提供明确入场点 + ML 过滤假信号 + 分形维度确保市场可交易
    缺点：三重过滤可能错过快速行情，PSAR 震荡市频繁翻转
    """
    name = "v145_ensemble_ml_fractal_dim_psar"
    warmup = 500
    freq = "4h"

    ens_period: int = 120
    fd_period: int = 60
    fd_threshold: float = 1.4
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._psar_val = None
        self._psar_dir = None
        self._vote = None
        self._fractal = None
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

        self._psar_val, self._psar_dir = psar(highs, lows)
        self._atr = atr(highs, lows, closes, period=14)

        # Build features for ensemble
        rsi_arr = rsi(closes, period=14)
        adx_arr = adx(highs, lows, closes, period=14)
        features = np.column_stack([rsi_arr, adx_arr, self._atr])
        self._vote, _ = ensemble_vote(closes, features, period=self.ens_period)

        self._fractal = fractal_dim(closes, period=self.fd_period)

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
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        psar_dir = self._psar_dir[i]
        vote_val = self._vote[i]
        fd_val = self._fractal[i]
        atr_val = self._atr[i]
        if np.isnan(psar_dir) or np.isnan(atr_val):
            return
        if np.isnan(vote_val):
            vote_val = 0.0
        if np.isnan(fd_val):
            fd_val = 1.5  # default to random walk

        self.bars_since_last_scale += 1

        # PSAR flip detection
        if i < 1:
            return
        prev_dir = self._psar_dir[i - 1]
        if np.isnan(prev_dir):
            return
        psar_flip_up = prev_dir == -1 and psar_dir == 1
        psar_flip_down = prev_dir == 1 and psar_dir == -1
        low_fractal = fd_val < self.fd_threshold

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
            if psar_flip_up and vote_val > 0 and low_fractal:
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
            elif psar_flip_down and vote_val < 0 and low_fractal:
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
                    and psar_dir == 1 and vote_val > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and psar_dir == -1 and vote_val < 0):
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
