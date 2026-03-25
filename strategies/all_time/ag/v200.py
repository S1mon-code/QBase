import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.ensemble_signal import ensemble_vote
from indicators.trend.adx import adx
from indicators.volume.mfi import mfi
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi
from indicators.momentum.roc import rate_of_change

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV200(TimeSeriesStrategy):
    """
    策略简介：ML集成投票(Ridge+Lasso+RF) + ADX趋势过滤 + MFI资金流的综合ML策略。

    使用指标：
    - Ensemble Vote(120): 三模型投票方向，agreement为一致性
    - ADX(14): 趋势强度过滤，>25确认趋势存在
    - MFI(14): 资金流指标，>50资金流入/<50资金流出
    - ATR(14): 止损距离计算

    进场条件（做多）：vote_score>0.3 + agreement>=0.67 + ADX>25 + MFI>50
    进场条件（做空）：vote_score<-0.3 + agreement>=0.67 + ADX>25 + MFI<50

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - vote_score方向反转 或 ADX<20

    优点：多模型集成降低单模型过拟合风险，ADX+MFI双重过滤
    缺点：三模型训练开销大，agreement阈值可能过滤太多信号
    """
    name = "ag_alltime_v200"
    warmup = 300
    freq = "4h"

    ens_period: int = 120         # Optuna: 80-160
    vote_thresh: float = 0.3      # Optuna: 0.2-0.5
    agree_thresh: float = 0.67    # Optuna: 0.5-1.0
    adx_period: int = 14          # Optuna: 10-20
    adx_threshold: float = 25.0   # Optuna: 18-35
    mfi_period: int = 14          # Optuna: 10-20
    atr_stop_mult: float = 3.0   # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._vote = None
        self._agree = None
        self._adx = None
        self._mfi = None
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

        # Build features for ensemble
        rsi_arr = rsi(closes, 14)
        adx_arr = adx(highs, lows, closes, self.adx_period)
        atr_arr = atr(highs, lows, closes, 14)
        roc_arr = rate_of_change(closes, 12)
        self._atr = atr_arr
        self._adx = adx_arr

        features = np.column_stack([rsi_arr, adx_arr, atr_arr, roc_arr])
        self._vote, self._agree = ensemble_vote(
            closes, features, period=self.ens_period)

        self._mfi = mfi(highs, lows, closes, volumes, self.mfi_period)

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
        vote = self._vote[i]
        agree = self._agree[i]
        adx_val = self._adx[i]
        mfi_val = self._mfi[i]

        if np.isnan(atr_val) or atr_val <= 0:
            return
        if np.isnan(vote) or np.isnan(agree) or np.isnan(adx_val) or np.isnan(mfi_val):
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

        # 3. Signal exit: vote reversal or ADX collapse
        if side == 1 and (vote < -self.vote_thresh or adx_val < 20):
            context.close_long()
            self._reset_state()
        elif side == -1 and (vote > self.vote_thresh or adx_val < 20):
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry: ensemble + ADX + MFI
        if side == 0:
            if (vote > self.vote_thresh and agree >= self.agree_thresh
                    and adx_val > self.adx_threshold and mfi_val > 50):
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
            elif (vote < -self.vote_thresh and agree >= self.agree_thresh
                    and adx_val > self.adx_threshold and mfi_val < 50):
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
            if self.direction == 1 and vote > self.vote_thresh and adx_val > self.adx_threshold:
                add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    self.bars_since_last_scale = 0
            elif self.direction == -1 and vote < -self.vote_thresh and adx_val > self.adx_threshold:
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
