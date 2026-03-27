import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.online_regression import online_sgd_signal
from indicators.momentum.tsi import tsi
from indicators.seasonality.weekday_effect import weekday_effect
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class OnlineTSIWeekdayStrategy(TimeSeriesStrategy):
    """
    策略简介：Online Learning（SGD）+ TSI动量 + 星期效应加权的1h多空策略。

    使用指标：
    - Online SGD Signal(0.01,20): 在线学习回归预测方向
    - TSI(25,13,7): 真实强度指数，TSI>0且>signal=多头动量
    - Weekday Effect(252): 历史星期效应，有利星期加权开仓
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Online SGD预测 > 0
    - TSI > 0 且 TSI > signal线
    - 当前星期效应 > 0（有利星期）

    进场条件（做空）：
    - Online SGD预测 < 0
    - TSI < 0 且 TSI < signal线
    - 当前星期效应 < 0（不利星期）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - TSI穿越零轴

    优点：在线学习自适应市场变化+TSI平滑动量+星期效应提供统计边际
    缺点：在线学习对噪音敏感，星期效应可能随市场结构变化失效
    """
    name = "v172_online_tsi_weekday"
    warmup = 800
    freq = "1h"

    tsi_long: int = 25
    tsi_short: int = 13
    tsi_signal: int = 7
    sgd_lr: float = 0.01
    sgd_period: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._sgd_pred = None
        self._tsi = None
        self._tsi_sig = None
        self._weekday = None
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
        self._tsi, self._tsi_sig = tsi(closes, long=self.tsi_long, short=self.tsi_short,
                                        signal=self.tsi_signal)

        # Online SGD features
        rets = np.zeros_like(closes)
        rets[1:] = closes[1:] / closes[:-1] - 1
        vol20 = np.full_like(closes, np.nan)
        for idx in range(20, len(closes)):
            vol20[idx] = np.std(rets[idx-20:idx])
        features = np.column_stack([rets, vol20])
        features = np.nan_to_num(features, nan=0.0)
        self._sgd_pred, _ = online_sgd_signal(closes, features,
                                               learning_rate=self.sgd_lr, period=self.sgd_period)

        # Weekday effect
        datetimes = context.get_full_datetime_array()
        self._weekday = weekday_effect(closes, datetimes, lookback=252)

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

        sgd_val = self._sgd_pred[i]
        tsi_val = self._tsi[i]
        tsi_sig = self._tsi_sig[i]
        wd_val = self._weekday[i]
        atr_val = self._atr[i]
        if np.isnan(sgd_val) or np.isnan(tsi_val) or np.isnan(tsi_sig) or np.isnan(atr_val):
            return
        if np.isnan(wd_val):
            wd_val = 0.0

        self.bars_since_last_scale += 1

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

        # ── 3. TSI穿越零轴退出 ──
        if side == 1 and tsi_val < 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and tsi_val > 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if sgd_val > 0 and tsi_val > 0 and tsi_val > tsi_sig and wd_val > 0:
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
            elif sgd_val < 0 and tsi_val < 0 and tsi_val < tsi_sig and wd_val < 0:
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
                    and tsi_val > 0 and tsi_val > tsi_sig):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and tsi_val < 0 and tsi_val < tsi_sig):
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
