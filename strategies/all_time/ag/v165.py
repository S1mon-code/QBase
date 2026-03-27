import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.ensemble_signal import ensemble_vote
from indicators.volatility.bollinger import bollinger_bands
from indicators.microstructure.amihud import amihud_illiquidity
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class EnsembleBBAmihudStrategy(TimeSeriesStrategy):
    """
    策略简介：ML Ensemble投票 + Bollinger Band突破 + Amihud流动性过滤的4h多空策略。

    使用指标：
    - Ensemble Vote(120): 3模型投票方向，vote>0=多数看多
    - Bollinger Bands(20,2.0): BB突破上轨=多头突破，突破下轨=空头突破
    - Amihud Illiquidity(20): 流动性过滤，过高不交易（滑点风险）
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Ensemble vote > 0（ML多数看多）
    - 价格突破BB上轨
    - Amihud < 中位数的2倍（流动性正常）

    进场条件（做空）：
    - Ensemble vote < 0（ML多数看空）
    - 价格跌破BB下轨
    - Amihud < 中位数的2倍（流动性正常）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - 价格回到BB中轨

    优点：多模型投票降低单模型风险+BB突破明确+流动性过滤减少滑点
    缺点：BB突破在假突破频繁市场效果差，ensemble计算开销大
    """
    name = "v165_ensemble_bb_amihud"
    warmup = 500
    freq = "4h"

    bb_period: int = 20
    bb_std: float = 2.0
    amihud_period: int = 20
    amihud_mult: float = 2.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._ens_vote = None
        self._bb_upper = None
        self._bb_mid = None
        self._bb_lower = None
        self._amihud = None
        self._amihud_median = None
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
        self._bb_upper, self._bb_mid, self._bb_lower = bollinger_bands(
            closes, period=self.bb_period, num_std=self.bb_std)
        self._amihud = amihud_illiquidity(closes, volumes, period=self.amihud_period)

        # Precompute rolling median of amihud for threshold
        n = len(closes)
        self._amihud_median = np.full(n, np.nan)
        window = 120
        for idx in range(window, n):
            valid = self._amihud[idx-window:idx]
            valid = valid[~np.isnan(valid)]
            if len(valid) > 0:
                self._amihud_median[idx] = np.median(valid)

        # ML features
        rets = np.zeros_like(closes)
        rets[1:] = closes[1:] / closes[:-1] - 1
        vol20 = np.full_like(closes, np.nan)
        for idx in range(20, len(closes)):
            vol20[idx] = np.std(rets[idx-20:idx])
        features = np.column_stack([rets, vol20])
        features = np.nan_to_num(features, nan=0.0)
        self._ens_vote, _ = ensemble_vote(closes, features, period=120)

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

        ens_val = self._ens_vote[i]
        bb_u = self._bb_upper[i]
        bb_m = self._bb_mid[i]
        bb_l = self._bb_lower[i]
        ami = self._amihud[i]
        ami_med = self._amihud_median[i]
        atr_val = self._atr[i]
        if np.isnan(ens_val) or np.isnan(bb_u) or np.isnan(atr_val):
            return
        if np.isnan(ami) or np.isnan(ami_med):
            ami_ok = True
        else:
            ami_ok = ami < ami_med * self.amihud_mult

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

        # ── 3. 回到BB中轨退出 ──
        if side == 1 and price < bb_m:
            context.close_long()
            self._reset_state()
        elif side == -1 and price > bb_m:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0 and ami_ok:
            if ens_val > 0 and price > bb_u:
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
            elif ens_val < 0 and price < bb_l:
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
                    and ens_val > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and ens_val < 0):
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
